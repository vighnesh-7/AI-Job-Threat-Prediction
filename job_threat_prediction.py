import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QRadioButton, QButtonGroup, QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
from PyQt5.QtCore import Qt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif



class StyledRadioButton(QRadioButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QRadioButton {
                padding: 5px;
                border-radius: 5px;
                margin: 2px 0;
                font-size: 13px;
            }
            QRadioButton:hover {
                background-color: #f0f0f0;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator::unchecked {
                border: 2px solid #999;
                background: white;
                border-radius: 9px;
            }
            QRadioButton::indicator::checked {
                border: 2px solid #2ecc71;
                background: #2ecc71;
                border-radius: 9px;
            }
        """)



class StyledRadioButton(QRadioButton):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet("""
            QRadioButton {
                padding: 5px;
                border-radius: 5px;
                margin: 2px 0;
                font-size: 13px;
            }
            QRadioButton:hover {
                background-color: #f0f0f0;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator::unchecked {
                border: 2px solid #999;
                background: white;
                border-radius: 9px;
            }
            QRadioButton::indicator::checked {
                border: 2px solid #2ecc71;
                background: #2ecc71;
                border-radius: 9px;
            }
        """)

class JobThreatPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
        self.original_df = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # File input
        self.file_btn = QPushButton('Select Dataset')
        self.file_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.file_btn.clicked.connect(self.load_file)
        left_layout.addWidget(self.file_btn)

        # Preprocessing techniques
        preprocess_group = QButtonGroup(self)
        self.normalize_btn = StyledRadioButton('Normalization')
        self.impute_btn = StyledRadioButton('Mean Imputation')
        self.discretize_btn = StyledRadioButton('Discretization')
        preprocess_group.addButton(self.normalize_btn)
        preprocess_group.addButton(self.impute_btn)
        preprocess_group.addButton(self.discretize_btn)

        # Update section labels style
        section_style = """
            QLabel {
                font-weight: bold;
                font-size: 16px;
                color: #2c3e50;
                margin-top: 15px;
            }
        """
        
        preprocessing_label = QLabel('Preprocessing Techniques:')
        preprocessing_label.setStyleSheet(section_style)
        left_layout.addWidget(preprocessing_label)
        left_layout.addWidget(self.normalize_btn)
        left_layout.addWidget(self.impute_btn)
        left_layout.addWidget(self.discretize_btn)

        # Classification algorithms
        classify_group = QButtonGroup(self)
        self.id3_btn = StyledRadioButton('ID3 (Decision Tree)')
        self.naive_bayes_btn = StyledRadioButton('Naive Bayes')
        classify_group.addButton(self.id3_btn)
        classify_group.addButton(self.naive_bayes_btn)
        classification_label = QLabel('Classification Algorithms:')
        classification_label.setStyleSheet(section_style)
        left_layout.addWidget(classification_label)
        left_layout.addWidget(self.id3_btn)
        left_layout.addWidget(self.naive_bayes_btn)
        left_layout.setSpacing(5)

        # Apply button
        self.apply_btn = QPushButton('Apply')
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        self.apply_btn.clicked.connect(self.apply_changes)
        left_layout.addWidget(self.apply_btn)

        # Data preview
        self.table = QTableWidget()
        right_layout.addWidget(QLabel('Data Preview:'))
        right_layout.addWidget(self.table)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
                background-color: #f8f9fa;
                font-size: 16px;
                line-height: 1.4;
            }
        """)
        right_layout.addWidget(QLabel('Results:'))
        right_layout.addWidget(self.results_text)
        self.results_text.setFixedHeight(300)


        # Create a splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        self.setWindowTitle('AI Job Threat Prediction System')
        self.setGeometry(100, 100, 1200, 800)

        # Update left layout spacing
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Update table style
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 5px;
                border: none;
                font-weight: bold;
            }
        """)

        # Update splitter style
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #e0e0e0;
                width: 2px;
            }
        """)
        
        # Set the main window style
        self.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'CSV Files (*.csv)')
        if filename:
            self.df = pd.read_csv(filename)
            self.original_df = self.df.copy()
            self.results_text.setText(f"Dataset loaded. Shape: {self.df.shape}")
            self.update_table()

    def update_table(self):
        if self.df is not None:
            self.table.setColumnCount(len(self.df.columns))
            self.table.setRowCount(len(self.df))
            self.table.setHorizontalHeaderLabels(self.df.columns)
            for i in range(len(self.df)):
                for j in range(len(self.df.columns)):
                    value = self.df.iloc[i, j]
                    if pd.isna(value):
                        item_text = "NaN"
                    else:
                        item_text = str(value)
                    self.table.setItem(i, j, QTableWidgetItem(item_text))
            self.table.resizeColumnsToContents()
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

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

            self.update_table()
        except Exception as e:
            self.results_text.append(f"An error occurred: {str(e)}")

    def normalize_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            scaler = MinMaxScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
            self.results_text.append("\nData normalized (excluding job_threat_level).")

    def impute_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy='mean')
            self.df[numeric_columns] = imputer.fit_transform(self.df[numeric_columns])
            self.results_text.append("\nMissing values imputed with mean for numeric columns (excluding job_threat_level).")

    def discretize_data(self):
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'job_threat_level']
        
        if len(numeric_columns) > 0:
            for column in numeric_columns:
                self.df[column] = pd.qcut(self.df[column], q=3, labels=['Low', 'Medium', 'High'])
            self.results_text.append("\nNumeric data discretized into 3 bins (excluding job_threat_level).")

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
            self.results_text.append("\nError: No valid data after removing NaN values.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results_text.append(f"\nID3 (Decision Tree) applied. Accuracy: {accuracy:.2f}")
        self.results_text.append(f"Number of samples used: {len(X)}\n")

        # Calculate information gain
        info_gains = mutual_info_classif(X, y)
        for feature, info_gain in zip(X.columns, info_gains):
            self.results_text.append(f"Information Gain for {feature}: {info_gain:.4f}")

    def apply_naive_bayes(self):
        X, y = self.prepare_data_for_classification()
        
        if len(X) == 0:
            self.results_text.append("\nError: No valid data after removing NaN values.")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.results_text.append(f"\nNaive Bayes applied. Accuracy: {accuracy:.2f}")
        self.results_text.append(f"Number of samples used: {len(X)}\n")

        # Calculate class probabilities
        class_probs = clf.class_prior_
        for class_label, prob in zip(clf.classes_, class_probs):
            self.results_text.append(f"Prior probability for class {class_label}: {prob:.4f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JobThreatPredictionApp()
    ex.show()
    sys.exit(app.exec_())

