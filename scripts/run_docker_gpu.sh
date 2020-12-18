#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

docker run \
  --name rlzoo3 \
  -it \
  --runtime=nvidia \
  --rm \
  --network host \
  --ipc=host \
  -v $(pwd)/../../gym-supplychain:/root/code/gym-supplychain \
  -v $(pwd)/../../phd-research/supply-chain/stochastic_demands/supplychain_ext1:/root/code/supplychain_ext1 \
  --mount src=$(pwd),target=/root/code/rl_zoo,type=bind \
  juliocaburu/rl-baselines3-zoo:latest \
  bash -c "cd /root/code/rl_zoo/ && bash"
