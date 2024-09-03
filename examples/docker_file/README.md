# Running a flyte workflow with a docker file
```bash
docker build . -t localhost:30000/test_docker:latest
docker push localhost:30000/test_docker:latest

# Init the local kubernetes cluster
export FLYTECTL_CONFIG=~/.flyte/config-sandbox.yaml
flytectl demo start
flytectl create project \
      --id "docker-file" \
      --labels "my-label=docker-file" \
      --description "A test with docker file" \
      --name "docker-file"

# Since the task is specified in the same file as the workflow, we have to install all the requirements to start the workflow 
pip install -r requirements.txt
cd workflows
pyflyte run --remote --image localhost:30000/test_docker:latest -p docker-file -d development example.py wf --name Gensim
```
