# USAGE
# python index_images.py --images 101_ObjectCategories --tree vptree.pickle --hashes hashes.pickle

# import the necessary packages
import time
from utils.hashing import convert_hash
from utils.hashing import hamming
from utils.hashing import dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, type=str, help="path to input directory of images")
ap.add_argument("-r", "--run", required=True, type=str, help="run name")
ap.add_argument("-l", "--height", type=int, default=200, help="image height")
ap.add_argument("-w", "--width", type=int, default=200, help="image width")

ap.add_argument("-q", "--query", type=str, help="path to input query image")
ap.add_argument("--dist", type=int, default=10, help="maximum hamming distance")

# ap.add_argument("-s", "--search", required=True, type=bool, default=True, help="path to input directory of images")

ap.add_argument('--search', dest='find', action='store_true')
ap.add_argument('--index', dest='find', action='store_false')
ap.set_defaults(find=True)

args = vars(ap.parse_args())

if args["find"] == True and args["query"]==None:
    ap.error('--query has to be set when --search')

OUTPUT_PATH = "output/" + str(args["run"]) + "/"
dim = (args["height"], args["width"])

try:
    os.mkdir(OUTPUT_PATH)
except:
    print("Directory already present.")
    pass

if __name__ == '__main__':

    if args["find"]:

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
        results = tree.get_all_in_range(queryHash, args["dist"])
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


    else:
        # grab the paths to the input images and initialize the dictionary
        # of hashes
        imagePaths = list(paths.list_images(args["data"]))

        hashes = {}
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # load the input image
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            raw_image = cv2.imread(imagePath)
            print('Original Dimensions : ', raw_image.shape)
            image = cv2.resize(raw_image, dim, interpolation=cv2.INTER_AREA)
            print('Resized Dimensions : ', image.shape)

            # compute the hash for the image and convert it
            h = dhash(image)
            h = convert_hash(h)

            # update the hashes dictionary
            l = hashes.get(h, [])
            l.append(imagePath)
            hashes[h] = l

        # build the VP-Tree
        print("[INFO] building VP-Tree...")
        points = list(hashes.keys())
        tree = vptree.VPTree(points, hamming)

        # serialize the VP-Tree to disk
        print("[INFO] serializing VP-Tree...")
        f = open(OUTPUT_PATH + "tree", "wb+")
        f.write(pickle.dumps(tree))
        f.close()

        # serialize the hashes to dictionary
        print("[INFO] serializing hashes...")
        f = open(OUTPUT_PATH + "hashes", "wb+")
        f.write(pickle.dumps(hashes))
        f.close()
