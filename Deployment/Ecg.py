from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
from scipy.signal import resample

class ECG:
    def getImage(self, image):
        """Load ECG image from file"""
        return imread(image)

    def GrayImage(self, image):
        """Convert image to grayscale and resize"""
        image_gray = color.rgb2gray(image)
        return resize(image_gray, (1572, 2213))

    def DividingLeads(self, image):
        """Divide ECG into 13 leads"""
        Lead_1 = image[300:600, 150:643]
        Lead_2 = image[300:600, 646:1135]
        Lead_3 = image[300:600, 1140:1625]
        Lead_4 = image[300:600, 1630:2125]
        Lead_5 = image[600:900, 150:643]
        Lead_6 = image[600:900, 646:1135]
        Lead_7 = image[600:900, 1140:1625]
        Lead_8 = image[600:900, 1630:2125]
        Lead_9 = image[900:1200, 150:643]
        Lead_10 = image[900:1200, 646:1135]
        Lead_11 = image[900:1200, 1140:1625]
        Lead_12 = image[900:1200, 1630:2125]
        Lead_13 = image[1250:1480, 150:2125]

        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7,
                 Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]

        # Plot 12 leads
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        x_counter = 0
        y_counter = 0
        for x, y in enumerate(Leads[:12]):
            ax[x_counter][y_counter].imshow(y, cmap='gray')
            ax[x_counter][y_counter].axis('off')
            ax[x_counter][y_counter].set_title(f"Lead {x+1}")
            y_counter += 1
            if y_counter == 3:
                x_counter += 1
                y_counter = 0
        fig.savefig('Leads_1-12_figure.png')

        # Plot long lead
        fig13, ax13 = plt.subplots(figsize=(10, 10))
        ax13.imshow(Lead_13, cmap='gray')
        ax13.set_title("Lead 13")
        ax13.axis('off')
        fig13.savefig('Long_Lead_13_figure.png')

        return Leads

    def PreprocessingLeads(self, Leads):
        """Preprocess leads using Gaussian blur + Otsu threshold"""
        fig2, ax2 = plt.subplots(4, 3, figsize=(10, 10))
        x_counter = 0
        y_counter = 0
        for x, y in enumerate(Leads[:12]):
            grayscale = color.rgb2gray(y)
            blurred_image = gaussian(grayscale, sigma=1)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            ax2[x_counter][y_counter].imshow(binary_global, cmap='gray')
            ax2[x_counter][y_counter].axis('off')
            ax2[x_counter][y_counter].set_title(f"Preprocessed Lead {x+1}")
            y_counter += 1
            if y_counter == 3:
                x_counter += 1
                y_counter = 0
        fig2.savefig('Preprocessed_Leads_1-12_figure.png')

        # Lead 13 preprocessing
        grayscale = color.rgb2gray(Leads[12])
        blurred_image = gaussian(grayscale, sigma=1)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        ax3.imshow(binary_global, cmap='gray')
        ax3.set_title("Preprocessed Lead 13")
        ax3.axis('off')
        fig3.savefig('Preprocessed_Lead_13_figure.png')

    def SignalExtraction_Scaling(self, Leads):
        """Extract 1D signal from leads, save contour plots, and scale to CSV"""
        
        # ---------- For Leads 1–12 ----------
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        x_counter, y_counter = 0, 0
        
        for idx, lead in enumerate(Leads[:12]):
            grayscale = color.rgb2gray(lead)
            blurred_image = gaussian(grayscale, sigma=0.7)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            
            contours = measure.find_contours(binary_global, 0.8)
            if not contours:
                continue
            
            largest_contour = max(contours, key=lambda x: x.shape[0])
            test = resize(largest_contour, (255, 2))
            
            # Save CSV
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(test)
            df_scaled = pd.DataFrame(scaled_data[:, 0]).T
            df_scaled.to_csv(f'Scaled_1DLead_{idx+1}.csv', index=False)
            
            # Plot contour
            ax[x_counter][y_counter].plot(test[:, 0], test[:, 1], linewidth=1)
            ax[x_counter][y_counter].invert_yaxis()
            ax[x_counter][y_counter].set_title(f"Contour Lead {idx+1}")
            ax[x_counter][y_counter].axis('off')
            
            y_counter += 1
            if y_counter == 3:
                x_counter += 1
                y_counter = 0
        
        fig.savefig('Contour_Leads_1-12_figure.png')
        plt.close(fig)

        # ---------- For Lead 13 ----------
        grayscale = color.rgb2gray(Leads[12])
        blurred_image = gaussian(grayscale, sigma=0.7)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        
        contours = measure.find_contours(binary_global, 0.8)
        if contours:
            largest_contour = max(contours, key=lambda x: x.shape[0])
            test = resize(largest_contour, (255, 2))
            
            # Save CSV for Lead 13
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(test)
            df_scaled = pd.DataFrame(scaled_data[:, 0]).T
            df_scaled.to_csv('Scaled_1DLead_13.csv', index=False)
            
            # Plot contour
            fig13, ax13 = plt.subplots(figsize=(10, 5))
            ax13.plot(test[:, 0], test[:, 1], linewidth=1)
            ax13.invert_yaxis()
            ax13.set_title("Contour Lead 13")
            ax13.axis('off')
            fig13.savefig('Contour_Lead_13_figure.png')
            plt.close(fig13)

    def CombineConvert1Dsignal(self):
        """Combine 12 lead 1D signals into one dataframe"""
        csv_files = [f for f in natsorted(os.listdir()) if f.startswith('Scaled_1DLead_') and f.endswith('.csv')]
        combined_df = pd.concat([pd.read_csv(f) for f in csv_files], axis=1, ignore_index=True)
        return combined_df

    def DimensionalReduciton(self, test_final):
        """Reduce dimensionality using PCA with resampling"""
        pca_loaded_model = joblib.load('pca.pkl')
        n_features = pca_loaded_model.components_.shape[1]
        # Resample rows to match PCA expected features
        resampled_data = np.array([resample(row, n_features) for row in test_final.values])
        result = pca_loaded_model.transform(resampled_data)
        return pd.DataFrame(result)

    def ModelLoad_predict(self, final_df):
        """Load trained model and predict ECG type"""
        loaded_model = joblib.load('best_trained_model.pkl')
        result = loaded_model.predict(final_df)
        mapping = {
            0: "Abnormal Heartbeat",
            1: "Myocardial Infarction",
            2: "Normal",
            3: "History of Myocardial Infarction"
        }
        return f"Your ECG corresponds to: {mapping.get(result[0], 'Unknown')}"

    def Save_LabelEncoder(self, le: LabelEncoder):
        """Save LabelEncoder"""
        joblib.dump(le, 'label_encoder.pkl')

    def Load_LabelEncoder(self):
        """Load LabelEncoder"""
        return joblib.load('label_encoder.pkl')
