def main():
    with open("fullNetwork.txt", "r") as file:
        string = file.read()

    arrays = string.split('][')

    simplifiedString = ''

    for i, array in enumerate(arrays):
        if i == 0:
            simplifiedString += str(array) + ']'
            continue

        weights = array.split(', ')

        simplifiedArary = []

        for weight in weights:
            if len(weight) > 2:
                simplifiedArary.append(float(weight[0:2]))
            else:
                simplifiedArary.append(float(weight))

        simplifiedString += str(simplifiedArary)

    with open("simplifiedNetwork.txt", "w") as file:
        file.write(simplifiedString)

if __name__ == '__main__':
    main()