import json

# Get json format data from the file via the input filename
def getdata(filename):
    print("Reading data from file: " + filename)
    data = json.load(open(filename))
    return data

