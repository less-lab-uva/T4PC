#!/bin/bash


##############
### Parser ###
##############

# Set default values
team_config=""
checkpoint_endpoint=""
save_path=""
tcp_output_type=""

# Function to display help
display_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --team_config <value>         Specify the team configuration"
    echo "  --checkpoint_endpoint <value> Specify the checkpoint endpoint"
    echo "  --save_path <value>           Specify the save path"
    echo "  --tcp_output_type <value>     Specify the TCP output signal to be send to CARLA"
    echo "  --help                        Display this help message"
}

# Check if no arguments are provided
if [ "$#" -eq 0 ]; then
    echo "Error: No arguments provided. Use --help for usage information."
    return
fi

# Process command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --team_config)
            shift
            # Check if a value is provided for --team_config
            if [ -n "$1" ]; then
                team_config="$1"
            else
                echo "Error: Value for --team_config is missing."
                return
            fi
            ;;
        --checkpoint_endpoint)
            shift
            # Check if a value is provided for --checkpoint_endpoint
            if [ -n "$1" ]; then
                checkpoint_endpoint="$1"
            else
                echo "Error: Value for --checkpoint_endpoint is missing."
                return
            fi
            ;;
        --save_path)
            shift
            # Check if a value is provided for --save_path
            if [ -n "$1" ]; then
                save_path="$1"
            else
                echo "Error: Value for --save_path is missing."
                return
            fi
            ;;
        --tcp_output_type)
            shift
            # Check if a value is provided for --tcp_output_type
            if [ -n "$1" ]; then
                tcp_output_type="$1"
            else
                echo "Error: Value for --tcp_output_type is missing."
                return
            fi
            ;;
        --help)
            display_help
            return
            ;;
        *)
            # Unknown option
            echo "Error: Unknown option '$1'. Use --help for usage information."
            return
            ;;
    esac
    # Move to the next argument
    shift
done

# Check if required parameters are provided
if [ -z "$team_config" ] || [ -z "$checkpoint_endpoint" ] || [ -z "$save_path" ] || [ -z "$tcp_output_type" ]; then
    echo "Error: all parameters --team_config, --checkpoint_endpoint, --save_path, and --tcp_output_type are required."
    return
fi

#####################
### End of Parser ###
#####################

source .env

module load gcc
module load libjpeg-turbo
module load cuda/11.4.2
module load jq

export CARLA_ROOT=$CARLA_DIR
export LEADERBOARD_ROOT=$LEADERBOARD_DIR

export CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$LEADERBOARD_ROOT
export PYTHONPATH=$PYTHONPATH:$LEADERBOARD_ROOT/team_code
export PYTHONPATH=$PYTHONPATH:$SCENARIO_RUNNER_DIR

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True

# TCP evaluation
export ROUTES=$LEADERBOARD_ROOT/data/evaluation_routes/routes_town05_long.xml
export SCENARIOS=$LEADERBOARD_ROOT/data/scenarios/town05_all_scenarios.json
export TEAM_AGENT=$LEADERBOARD_ROOT/team_code/tcp_agent.py
export TEAM_CONFIG=$team_config
export CHECKPOINT_ENDPOINT=$save_path/$checkpoint_endpoint
export SAVE_PATH=$save_path
export PYTHONUNBUFFERED=1
export TCP_OUTPUT_TYPE=$tcp_output_type

# Define port range
min_port=2000
max_port=2500

# Function to generate a random port within the range
get_random_port() {
  echo $((min_port + RANDOM % (max_port - min_port + 2)))
}

# Function to check if port is in use
is_port_in_use() {
  netstat -tuln | grep -q ":$1 "
}

# Function to check if both the given port and the next port (+1) are in use
are_ports_in_use() {
  is_port_in_use $1 || is_port_in_use $(( $1 + 1 ))
}

executeRoute() {    
    # Find an unused port for CARLA
    echo "Finding an unused port"
    port=$(get_random_port)
    while are_ports_in_use $port; do
        port=$(get_random_port)
    done
    export PORT=$port

    # Find an unused port for CARLA Traffic Manager
    tm_port=$((port + 6000))
    while is_port_in_use $tm_port; do
        tm_port=$((tm_port + 1))
    done
    export TM_PORT=$tm_port

    echo "Initializing CARLA on port $PORT"
    echo "Initializing CARLA Traffic Manager on port $TM_PORT"
    # Step 1: Get GPU usage. This command lists GPUs by their utilization, picking the first (least used) GPU.
    GPU_ID=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -n 1 | cut -d, -f1)
    # Step 2: Export the GPU ID to use it for the CARLA simulation.
    export SDL_HINT_CUDA_DEVICE=$GPU_ID
    ${CARLA_ROOT}/CarlaUE4.sh --world-port=$PORT -opengl &
    sleep 15

    python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --scenarios=${SCENARIOS}  \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --record=${RECORD_PATH} \
    --resume=${RESUME} \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT} \
    --output_type=${TCP_OUTPUT_TYPE}

    pid=$(lsof -i :$PORT | awk 'NR==2 {print $2}')
    echo "Killing CARLA on port $PORT"
    kill -9 $pid
    pid2=$(lsof -i :$TM_PORT | awk 'NR==2 {print $2}')
    echo "Killing CARLA Traffic Manager on port $TM_PORT"
    kill -9 $pid2
    sleep 5
}

finalreport() {
    # Find an unused port for CARLA
    echo "Finding an unused port"
    port=$(get_random_port)
    while are_ports_in_use $port; do
        port=$(get_random_port)
    done
    export PORT=$port

    # Find an unused port for CARLA Traffic Manager
    tm_port=$((port + 6000))
    while is_port_in_use $tm_port; do
        tm_port=$((tm_port + 1))
    done
    export TM_PORT=$tm_port

    echo "Initializing CARLA on port $PORT"
    # Step 1: Get GPU usage. This command lists GPUs by their utilization, picking the first (least used) GPU.
    GPU_ID=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | sort -t, -k2 -n | head -n 1 | cut -d, -f1)
    # Step 2: Export the GPU ID to use it for the CARLA simulation.
    export SDL_HINT_CUDA_DEVICE=$GPU_ID
    ${CARLA_ROOT}/CarlaUE4.sh --world-port=$PORT -opengl &
    sleep 5

    python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --scenarios=${SCENARIOS}  \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --record=${RECORD_PATH} \
    --resume=${RESUME} \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT} \
    --done

    pid=$(lsof -i :$PORT | awk 'NR==2 {print $2}')
    echo "Killing CARLA on port $PORT"
    kill -9 $pid
    pid2=$(lsof -i :$TM_PORT | awk 'NR==2 {print $2}')
    echo "Killing CARLA Traffic Manager on port $TM_PORT"
    kill -9 $pid2
    sleep 5
}

sleep 2
mkdir -p $save_path
sleep 2
flag=true

while $flag; do
    json_exists=$(find $save_path -name results.json)
    if [ -z "$json_exists" ]; then
        executeRoute
    fi
    progress=($(cat $save_path/results.json | jq '._checkpoint.progress[]')) || { echo "Command failed"; exit 1; }
    if [ ${progress[0]} -ne ${progress[1]} ]; then
        executeRoute
    else
        flag=false
    fi
done

echo "Final report"
finalreport