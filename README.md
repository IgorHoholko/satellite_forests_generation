# satellite_forests_generation
Generating forests on satellite images


## Pre-Install (If you want to use GPU)
1) [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
2) [Install Nvidia Docker Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
3) Set `"default-runtime" : "nvidia"` in `/etc/docker/daemon.json`:
    ```json
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    ```
4) Restart Docker:
``` bash
sudo systemctl restart docker
```
