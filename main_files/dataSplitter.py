import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, dataset_path, save_dir):
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Define the variations and columns to drop for each
        self.variations = {
            "with_gender_and_age": [],
            "gender_no_age": [4, 5],           # Exclude age and keep gender
            "age_no_gender": [3],              # Exclude gender and keep age
            "no_age_no_gender": [3, 4, 5],     # Exclude both age and gender
        }

    def load_dataset(self):
        """Load the full dataset."""
        df = pd.read_csv(self.dataset_path, header=None)
        return df

    def save_variation(self, df, prefix, columns_to_drop):
        """Save the variation of the dataset, dropping specified columns."""
        df_selected = df.drop(columns=columns_to_drop, axis=1) if columns_to_drop else df.copy()
        save_path = os.path.join(self.save_dir, f"{prefix}.csv")
        df_selected.to_csv(save_path, index=False, header=False)
        print(f"Full dataset with {prefix.replace('_', ' and ')} saved to {save_path}")
        return df_selected

    def split_and_save(self, df, prefix):
        """Split the dataset into train and test sets and save them."""
        train_vectors, test_vectors = train_test_split(df.values, test_size=0.20, random_state=42)
        
        # Convert back to DataFrame for saving
        df_train = pd.DataFrame(train_vectors)
        df_test = pd.DataFrame(test_vectors)

        df_train.to_csv(os.path.join(self.save_dir, f"train_{prefix}.csv"), index=False, header=False)
        df_test.to_csv(os.path.join(self.save_dir, f"test_{prefix}.csv"), index=False, header=False)
        print(f"Train and test sets for {prefix.replace('_', ' and ')} saved.")

    def process(self):
        """Process the dataset: create variations and split into train/test."""
        # Load the full dataset
        full_df = self.load_dataset()
        
        # Iterate over each variation
        for variation, columns_to_drop in self.variations.items():
            # Save the variation of the full dataset
            df_variation = self.save_variation(full_df, f"full_{variation}", columns_to_drop)
            
            # Split and save train and test sets for each variation
            self.split_and_save(df_variation, variation)


# Usage
def main():
    # Path to the full dataset (assumed to be in CSV format)
    dataset_path = "datasets/employes_flat_version.csv"
    save_dir = "datasets"

    # Create an instance of the DataSplitter and process the dataset
    splitter = DataSplitter(dataset_path, save_dir)
    splitter.process()

if __name__ == "__main__":
    main()
