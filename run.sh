task_id="webgen"
model_name="llama3.3-70b"
prompt_name="default"

uv run -m src.run_task \
    --task_id=${task_id} \
    --model=${model_name} \
    --prompt_name=${prompt_name} \
    --max_examples=5

data_path="results/${task_id}/${model_name}_${prompt_name}.json"

uv run -m src.evaluate \
    --task_id=${task_id} \
    --data_path=${data_path}