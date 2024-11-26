import os
import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt5.QtCore import Qt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "xcb"

class DataPreviewWindow(QWidget):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        table = QTableWidget()
        table.setColumnCount(len(self.df.columns))
        table.setRowCount(len(self.df))
        
        table.setHorizontalHeaderLabels(self.df.columns)
        
        for i in range(len(self.df)):
            for j in range(len(self.df.columns)):
                value = self.df.iloc[i, j]
                if pd.isna(value):
                    item_text = "NaN"
                else:
                    item_text = str(value)
                table.setItem(i, j, QTableWidgetItem(item_text))
        
        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(table)
        self.setLayout(layout)
        self.setWindowTitle('Data Preview')
        self.setGeometry(300, 300, 800, 600)

class JobThreatPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.df = None
        self.original_df = None

    def initUI(self):
        layout = QVBoxLayout()

        self.file_btn = QPushButton('Select Dataset')
        self.file_btn.clicked.connect(self.load_file)
        layout.addWidget(self.file_btn)

        preprocess_group = QButtonGroup(self)
        self.normalize_btn = QRadioButton('Normalization')
        self.impute_btn = QRadioButton('Mean Imputation')
        self.discretize_btn = QRadioButton('Discretization')
        preprocess_group.addButton(self.normalize_btn)
        preprocess_group.addButton(self.impute_btn)
        preprocess_group.addButton(self.discretize_btn)
        layout.addWidget(QLabel('Preprocessing Techniques:'))
        layout.addWidget(self.normalize_btn)
        layout.addWidget(self.impute_btn)
        layout.addWidget(self.discretize_btn)

        classify_group = QButtonGroup(self)
        self.id3_btn = QRadioButton('ID3 (Decision Tree)')
        self.naive_bayes_btn = QRadioButton('Naive Bayes')
        classify_group.addButton(self.id3_btn)
        classify_group.addButton(self.naive_bayes_btn)
        layout.addWidget(QLabel('Classification Algorithms:'))
        layout.addWidget(self.id3_btn)
        layout.addWidget(self.naive_bayes_btn)

        self.apply_btn = QPushButton('Apply')
        self.apply_btn.clicked.connect(self.apply_changes)
        layout.addWidget(self.apply_btn)

        self.preview_btn = QPushButton('Preview Data')
        self.preview_btn.clicked.connect(self.preview_data)
        layout.addWidget(self.preview_btn)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel('Results:'))
        layout.addWidget(self.results_text)

        self.setLayout(layout)
        self.setWindowTitle('AI Job Threat Prediction System')
        self.setGeometry(300, 300, 400, 500)

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv)')
        if filename:
            self.df = pd.read_csv(filename)
            self.original_df = self.df.copy()
            self.results_text.setText(f"Dataset loaded. Shape: {self.df.shape}")
            self.preview_btn.setEnabled(True)

    def apply_changes(self):
        if self.df is None:
            self.results_text.setText("Please load a dataset first.")
            return

        try:
            self.df = self.original_df.copy()
            
            if self.normalize_btn.isChecked():
                self.normalize_data()
            elif self.impute_btn.isChecked():
                self.impute_data()
            elif self.discretize_btn.isChecked():
                self.discretize_data()

            if self.id3_btn.isChecked():
                self.apply_id3()
            elif self.naive_bayes_btn.isChecked():
                self.apply_naive_bayes()
        except Exception as e:
            self.results_text.append(f"An error occurred: {str(e)}")

    def normalize_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            scaler = MinMaxScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
            self.results_text.append("Data normalized (excluding job_threat_level).")

    def impute_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy='mean')
            self.df[numeric_columns] = imputer.fit_transform(self.df[numeric_columns])
            self.results_text.append("Missing values imputed with mean for numeric columns (excluding job_threat_level).")

    def discretize_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            for column in numeric_columns:
                self.df[column] = pd.qcut(self.df[column], q=3, labels=['Low', 'Medium', 'High'])
            self.results_text.append("Numeric data discretized into 3 bins (excluding job_threat_level).")

    def prepare_data_for_classification(self):
        df_clean = self.df.dropna(subset=['job_threat_level'])
        
        X = df_clean.drop('job_threat_level', axis=1)
        y = df_clean['job_threat_level']
        
        X = pd.get_dummies(X)
        X = X.fillna(X.mean())
        
        return X, y

    def apply_id3(self):
        X, y = self.prepare_data_for_classification()
        
        if len(X) == 0:
            self.results_text.append("Error: No valid data after removing NaN values.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results_text.append(f"ID3 (Decision Tree) applied. Accuracy: {accuracy:.2f}")
        self.results_text.append(f"Number of samples used: {len(X)}")

        # Calculate information gain
        info_gains = mutual_info_classif(X, y)
        for feature, info_gain in zip(X.columns, info_gains):
            self.results_text.append(f"Information Gain for {feature}: {info_gain:.4f}")

    def apply_naive_bayes(self):
        X, y = self.prepare_data_for_classification()
        
        if len(X) == 0:
            self.results_text.append("Error: No valid data after removing NaN values.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results_text.append(f"Naive Bayes applied. Accuracy: {accuracy:.2f}")
        self.results_text.append(f"Number of samples used: {len(X)}")

        # Calculate class probabilities
        class_probs = clf.class_prior_
        for class_label, prob in zip(clf.classes_, class_probs):
            self.results_text.append(f"Prior probability for class {class_label}: {prob:.4f}")

        # Calculate feature probabilities for each class
        feature_probs = clf.theta_
        for i, class_label in enumerate(clf.classes_):
            self.results_text.append(f"\nFeature probabilities for class {class_label}:")
            for j, feature in enumerate(X.columns):
                self.results_text.append(f"{feature}: mean={clf.theta_[i,j]:.4f}, var={clf.sigma_[i,j]:.4f}")

    def preview_data(self):
        if self.df is not None:
            self.preview_window = DataPreviewWindow(self.df)
            self.preview_window.show()
        else:
            self.results_text.setText("Please load a dataset first.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JobThreatPredictionApp()
    ex.show()
    sys.exit(app.exec_())


