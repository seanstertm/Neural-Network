import json
import requests

def main():
    data = requests.get("https://raw.githubusercontent.com/SebLague/Neural-Network-Experiments/main/Assets/Data/Mnist/Trained%20Networks/Mnist%20Net.json").json()
    network = str(data["layerSizes"])
    network += str(data["connections"][0]["weights"])
    network += str(data["connections"][0]["biases"])
    network += str(data["connections"][1]["weights"])
    network += str(data["connections"][1]["biases"])
    with open("network.txt", "w") as file:
        file.write(network)



if __name__ == '__main__':
    main()