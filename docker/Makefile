repo_tag=federicociner

build_dir=latest
image_name=cs7641-a2
image_tag=$(image_name):$(build_dir)

full_image_tag=$(repo_tag)/$(image_tag)

buildlocal:
	@echo "Building container"
	sudo docker build $(build_dir) --tag $(full_image_tag)

build:
	sudo docker build $(build_dir)

runlocal:
	@echo "Running container locally"
	sudo docker run -t -d --name $(image_name) -v ~/Projects:/home/Projects:z $(full_image_tag) /bin/bash

runbash:
	docker exec -it $(image_name) /bin/bash -c "export COLUMNS=`tput cols`; export LINES=`tput lines`; exec bash"

.PHONY: buildlocal build runlocal

