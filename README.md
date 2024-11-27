# OpenRamanDatabase

OpenRamanDatabase is a software tool designed to analyze data from an open Raman spectroscope and match it against a database of known microplastics. It was built to be associated with a lowtech/cost raman spectroscope. Thus a simple peak matching algorithm and a threshold adjustment for cutting out noise.
![image](https://github.com/user-attachments/assets/ceff6078-6096-4b70-97d6-e133583c0fec)
![image](https://github.com/user-attachments/assets/9d7fb720-5e17-47dd-aabd-c0e040c956e5)

## Prerequisites

Before you start, ensure you have Docker and Docker Compose installed on your machine.

### Installing Docker

1. **For Windows and Mac:**
   - Download Docker Desktop from the [Docker Official Site](https://www.docker.com/products/docker-desktop).
   - Follow the installation instructions provided on the site.

2. **For Linux:**
   - You can install Docker from the terminal using your package manager.
   - Example for Ubuntu:
     ```bash
     sudo apt-get update
     sudo apt-get install docker-ce docker-ce-cli containerd.io
     ```

### Installing Docker Compose

- Docker Compose is usually bundled with Docker Desktop for Windows and Mac.
- For Linux, you can install it via the following command:
  ```bash
  sudo curl -L "https://github.com/docker/compose/releases/download/$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose

### Installing the program

     ```bash
    git clone https://github.com/Sailowtech/OpenRamanDatabase.git
    cd OpenRamanDatabase
     ```
You can aslo download as zip and cd into the directory containing the dockerfile.

### Launching 

     ```bash
    docker compose build
    docker compose up
     ```
