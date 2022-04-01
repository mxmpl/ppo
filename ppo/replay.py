"""Provides a function to build the replay buffer using reverb.
It is similar to acme.agents.replay.make_reverb_online_queue, but
in the adder two parameters differ:
```
    pad_end_of_episode=False
    break_end_of_episode=False
```
and in the dataset, we set `max_in_flight_samples_per_worker=2*batch_size`
instead of 1.
"""
from collections import namedtuple
from typing import Any, Dict

import reverb
from acme import specs
from acme.adders import reverb as adders

EXTRA_CAPACITY: int = 1000

ReplayBuffer = namedtuple('ReplayBuffer',
                          ('data_iterator, server, adder, client, can_sample'))


def make_replay_buffer(
    environment_spec: specs.EnvironmentSpec,
    extra_spec: Dict[str, Any],
    sequence_length: int,
    batch_size: int,
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
) -> ReplayBuffer:
    """Make the ReplayBuffer.

    Parameters
    ----------
    environment_spec : specs.EnvironmentSpec
        Environment specifications.
    extra_spec : Dict[str, Any]
        Extra specifications, ie log probabilities of the actions.
    sequence_length : int
        Sequence length.
    batch_size : int
        Batch size.
    replay_table_name : str
        Replay table name, by default adders.DEFAULT_PRIORITY_TABLE.

    Returns
    -------
    ReplayBuffer
        Reverb replay buffer.
    """
    signature = adders.SequenceAdder.signature(
        environment_spec,
        extra_spec,
        sequence_length=sequence_length)
    queue = reverb.Table.queue(
        name=replay_table_name,
        max_size=batch_size+EXTRA_CAPACITY,
        signature=signature)
    server = reverb.Server([queue], port=None)

    client = reverb.Client(f'localhost:{server.port}')

    def can_sample(): return queue.can_sample(batch_size)

    address = f'localhost:{server.port}'
    adder = adders.SequenceAdder(
        client=reverb.Client(address),
        period=sequence_length-1,
        sequence_length=sequence_length,
        pad_end_of_episode=False,
        break_end_of_episode=False,
    )

    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=address,
        table=replay_table_name,
        max_in_flight_samples_per_worker=2*batch_size,
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    data_iterator = dataset.as_numpy_iterator()
    return ReplayBuffer(data_iterator, server, adder, client, can_sample)
