for env in 'Humanoid-v3'
do
    for seed in 0 100 200 300 400
    do
        python train_pdo_mujoco.py --seed ${seed} --env ${env}
    done
done