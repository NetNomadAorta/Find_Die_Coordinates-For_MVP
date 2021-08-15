# import the necessary packages
import os
import glob
import cv2
import time

# User Parameters/Constants to Set
MATCH_CL = 0.70 # Minimum confidence level (CL) required to match golden-image to scanned image
SPLIT_MATCHES_CL =  0.85 # Splits MATCH_CL to SPLIT_MATCHES_CL (defects) to one folder, rest (no defects) other folder
STICHED_IMAGES_DIRECTORY = "Images/Stitched_Images/"
GOLDEN_IMAGES_DIRECTORY = "Images/Golden_Images/"
SLEEP_TIME = 0.0 # Time to sleep in seconds between each window step


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


def slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # Slide a window across the resized full image
    for y in range(0, fullImage.shape[0], stepSizeY):
        for x in range(0, fullImage.shape[1], stepSizeX):
            # Yield the current window
            yield (x, y, fullImage[y:y + windowSize[1], x:x + windowSize[0]])


# Comparison scan of scanning window-image to golden-image
def getMatch(window, goldenImage, x, y):
    h1, w1, c1 = window.shape
    h2, w2, c2 = goldenImage.shape
    
    if c1 == c2 and h2 <= h1 and w2 <= w1:
        method = eval('cv2.TM_CCOEFF_NORMED')
        res = cv2.matchTemplate(window, goldenImage, method)   
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val > MATCH_CL: 
            print("\nFOUND MATCH")
            print("max_val = ", max_val)
            print("Coordinates: x1:", x + max_loc[0], "y1:", y + max_loc[1], \
                  "x2:", x + max_loc[0] + w2, "y2:", y + max_loc[1] + h2)
            
            # Gets coordinates of cropped image
            return (max_loc[0], max_loc[1], max_loc[0] + w2, max_loc[1] + h2, max_val)
        
        else:
            return ("null", "null", "null", "null", "null")


# MAIN():
# =============================================================================
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

# Deletes contents in cropped and split folders
deleteDirContents("./Images/Cropped_Die_Images/")

# load the full and comparing crop images
fullImagePath = glob.glob(STICHED_IMAGES_DIRECTORY + "*")
fullImage = cv2.imread(fullImagePath[0])
goldenImagePath = glob.glob(GOLDEN_IMAGES_DIRECTORY + "*")
goldenImage = cv2.imread(goldenImagePath[0])

# Parameter set
winW = round(goldenImage.shape[1] * 1.5) # Scales window width according to full image resolution
winH = round(goldenImage.shape[0] * 1.5) # Scales window height according to full image resolution
windowSize = (winW, winH)
stepSizeX = round(winW / 2.95)
stepSizeY = round(winH / 2.95)

# Predefine next for loop's parameters 
prev_y1 = stepSizeY * 9 # Number that prevents y = 0 = prev_y1
prev_x1 = stepSizeX * 9
prev_y2 = 0
prev_x2 = 0
prev_matchedCL = 0


# loop over the sliding window
for (x, y, window) in slidingWindow(fullImage, stepSizeX, stepSizeY, windowSize):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue
    
    # Draw rectangle over sliding window for debugging and easier visual
    clone = fullImage.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 180), 30)
    cloneResize = cv2.resize(clone, (round(fullImage.shape[1] / fullImage.shape[0] * 950), 950))
    cv2.imshow("Window", cloneResize)
    cv2.waitKey(1)
    time.sleep(SLEEP_TIME) # sleep time in ms after each window step
    
    # Scans window for matched image
    # ==================================================================================
    # Scans window and grabs cropped image coordinates relative to window
    # Uses each golden image in the file if multiple part types are present
    for goldenImagePath in glob.glob(GOLDEN_IMAGES_DIRECTORY + "*"):
        goldenImage = cv2.imread(goldenImagePath)
        win_x1, win_y1, win_x2, win_y2, matchedCL = getMatch(window, goldenImage, x, y)
        
        # Saves cropped image and names with coordinates
        if win_x1 != "null":
            # Turns cropped image coordinates relative to window to full image coordinates
            x1 = x + win_x1
            y1 = y + win_y1
            x2 = x + win_x2
            y2 = y + win_y2
            
            # Makes sure same image does not get saved as different names
            if y1 >= (prev_y1 + round(stepSizeY / 2.95) ) or y1 <= (prev_y1 - round(stepSizeY / 2.95)):
                sameCol = False
            else:
                if x1 >= (prev_x1 + round(stepSizeX / 2.95) ) or x1 <= (prev_x1 - round(stepSizeX / 2.95)):
                    prev_matchedCL = 0
                    sameCol = False
                else: 
                    sameCol = True
            
            if (sameCol == False) or (sameCol == True and matchedCL > prev_matchedCL): 
                # Gets cropped image and saves cropped image
                croppedImage = window[win_y1:win_y2, win_x1:win_x2]
                cv2.imwrite("./Images/Cropped_Die_Images/x1_{}-y1_{}-x2_{}-y2_{}.jpg".format(x1, y1, x2, y2), croppedImage)
                # If previous same Row and Column will be saved twice, deletes first one
                if sameCol == True and matchedCL > prev_matchedCL:
                    if "x1:{}-y1:{}-x2:{}-y2:{}.jpg".format(x1, y1, x2, y2) in os.listdir("./Images/Cropped_Die_Images/"): 
                        os.remove("./Images/Cropped_Die_Images/x1_{}-y1_{}-x2_{}-y2_{}.jpg".format(prev_x1, prev_y1, prev_x2, prev_y2), croppedImage)
            
            prev_y1 = y1
            prev_x1 = x1
            prev_y2 = y2
            prev_x2 = x2
            prev_matchedCL = matchedCL
        # ==================================================================================