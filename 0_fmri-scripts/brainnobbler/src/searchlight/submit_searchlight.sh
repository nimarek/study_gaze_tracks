logs_dir=/home/data/NegativeCueing_RSA/NegCue_Random/code/logs_sl-brainnobbler/
# create the logs dir if it doesn't exist
[ ! -d "$logs_dir" ] && mkdir -p "$logs_dir"

# exclude bad nodes from analysis

printf "# The environment
universe       = vanilla
getenv         = True
request_cpus   = 8
request_memory = 10G
# Execution
initial_dir    = /home/data/NegativeCueing_RSA/NegCue_Random/code/brainnobbler/src/first_level
executable     = searchlight.sh
\n"

for sub in 0{1..4} 0{6..9}; do
    printf "arguments = ${sub}\n"
    printf "log       = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).log\n"
    printf "output    = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).out\n"
    printf "error     = ${logs_dir}/sub-${sub}_\$(Cluster).\$(Process).err\n"
    printf "Queue\n\n"
done