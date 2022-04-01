workdir='walker'
for seed in `seq 1 10`; do
    python run.py \
    $seed \
    --horizon 4 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --num_minibatches 8 \
    --minibatch_size 8 \
    --discount 0.99 \
    --gae_lambda 0.95 \
    --clipping_epsilon 0.2 \
    --entropy_coeff 0 \
    --value_coeff 1 \
    --adam_epsilon 1e-5 \
    --max_gradient_norm 0.5 \
    --workdir $workdir \
    --num_steps 1_000_000 \
    --env walker \
; done