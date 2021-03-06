import sys
import pandas as pd
import numpy as np


OUTPUT_PATH = "data/labelled.csv"

def main(input_path):
    # read data here
    df = pd.read_csv(input_path)

    # IMPLEMENT DATA TRANSFORMATION HERE (IF REQUIRED)

    # IMPLEMENT MODEL LOADING HERE

    # IMPLEMENT MODEL INFERENCE HERE

    # ------ DUMMY TRANSFORMATION -------

    possible_predictions = ["happy", "sad", "neutral", "fury"]
    np.random.seed = 1
    dummy_predictions = np.random.choice(a=possible_predictions,size=len(df))
    dummy_predictions_df = pd.DataFrame(dummy_predictions)
    
    # --- END OF DUMMY TRANSFORMATION ---

    # SAVE PREDICTIONS TO CSV
    dummy_predictions_df.to_csv(OUTPUT_PATH, index=False, header=False)

if __name__ == "__main__":
    input_path = sys.argv[1]
    main(input_path)