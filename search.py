# USAGE
# python search.py --tree vptree.pickle --hashes hashes.pickle --query queries/accordion.jpg

# import the necessary packages
from utils.hashing import convert_hash
from utils.hashing import dhash
import argparse
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--run", required=True, type=str, help="path to pre-constructed variables")
ap.add_argument("-q", "--query", required=True, type=str, help="path to input query image")
ap.add_argument("-d", "--distance", type=int, default=10, help="maximum hamming distance")
ap.add_argument("-l", "--height", type=int, default=200, help="image height")
ap.add_argument("-w", "--width", type=int, default=200, help="image width")
args = vars(ap.parse_args())

OUTPUT_PATH = "output/" + str(args["run"]) + "/"

dim = (args["height"], args["width"])

# load the VP-Tree and hashes dictionary
print("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open(OUTPUT_PATH + "tree", "rb").read())
hashes = pickle.loads(open(OUTPUT_PATH + "hashes", "rb").read())

# load the input query image
raw_image = cv2.imread(args["query"])

image = cv2.resize(raw_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Query", image)

# compute the hash for the query image, then convert it
queryHash = dhash(image)
queryHash = convert_hash(queryHash)

# perform the search
print("[INFO] performing search...")
start = time.time()
results = tree.get_all_in_range(queryHash, args["distance"])
results = sorted(results)
end = time.time()
print("[INFO] search took {} seconds".format(end - start))
print(results)
# loop over the results
for (d, h) in results:
    # grab all image paths in our dataset with the same hash
    resultPaths = hashes.get(h, [])
    print("[INFO] {} total image(s) with d: {}, h: {}".format(
        len(resultPaths), d, h))

    # loop over the result paths
    for resultPath in resultPaths:
        # load the result image and display it to our screen

        print(resultPath)

        # result = cv2.imread(resultPath)
        # cv2.imshow("Result", result)
        # cv2.waitKey(0)
