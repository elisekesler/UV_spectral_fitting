import sys
import csv
from functools import partial
import numpy as np
from scipy.integrate import simpson
from astropy.table import vstack
from astropy.io import fits
import lmfit
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QHBoxLayout, QLineEdit, QProgressBar,
    QInputDialog, QMessageBox, QComboBox, QTableWidget,
    QTableWidgetItem, QSizePolicy, QFrame
)
from astropy.stats import poisson_conf_interval as pcf
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from prepare_spec import coadd, prepare, prepare_other_grating, combine_tables
import pickle

class SpectralFluxApp(QMainWindow):
    # Main app with all the doohickeys and functionalities
    def __init__(self, csv_file_path=None):
        super().__init__()

        self.init_variables()
        self.create_main_widget()
        self.create_plot_area()
        self.create_controls()
        self.create_table()
        self.create_right_panel()  # Add this line

        self.setWindowTitle('Spectral Flux Measurement')
        self.setGeometry(100, 100, 1200, 800)  # Make the window a bit wider

        # self.setWindowTitle('Spectral Flux Measurement')
        # self.setGeometry(100, 100, 1000, 800)

    def init_variables(self):
        self.spectrum_data = []
        self.grating_data = []
        self.coadded_spectrum = None
        self.fits_file_paths = []
        self.redshift = 0.0
        self.selection_step = 0
        self.cid = None  # Connection ID for event handler
        self.scale_value = 1e15 #value to scale the flux by
        self.flux_method = "Direct Integration"

        self.line_wavelengths = np.array([0.0,
            1215.67, 1031.92, 1037.61, 1238.82, 1242.8,
            1393.75, 1402.77, 1548.19, 1550.77
        ])
        self.line_labels = [
            'Full Spectrum', 
            'Lya', 'O VI 1031', 'O VI 1037', 'N V 1238',
            'N V 1242', 'Si 1393', 'Si 1402', 'C IV 1548', 'C IV 1550'
        ]

        # Variables for plotting selection overlays
        self.left_continuum_lines = []
        self.left_continuum_patch = None
        self.right_continuum_lines = []
        self.right_continuum_patch = None
        self.integration_lines = []
        self.integration_patch = None

    def create_main_widget(self):
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        # Create a horizontal layout for the entire application
        self.app_layout = QHBoxLayout(self.main_widget)
        
        # Create a vertical layout for the existing components (left side)
        self.main_layout = QVBoxLayout()
        self.app_layout.addLayout(self.main_layout)
        
        # Create a vertical layout for the new components (right side)
        self.right_layout = QVBoxLayout()
        self.app_layout.addLayout(self.right_layout)
        
        # Set the size ratio between left and right panels (2:1)
        self.app_layout.setStretch(0, 2)
        self.app_layout.setStretch(1, 1)

    def create_plot_area(self):
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.main_layout.addWidget(self.toolbar)

    def create_controls(self):
        self.controls_layout = QHBoxLayout()
        self.main_layout.addLayout(self.controls_layout)

        # Load data button
        self.load_button = QPushButton('Load CSV with FITS Paths')
        self.load_button.clicked.connect(self.load_csv_with_fits_paths)
        self.controls_layout.addWidget(self.load_button)

        # # Co-add and plot button
        # self.coadd_button = QPushButton('Co-add and Plot')
        # self.coadd_button.clicked.connect(self.coadd_and_plot)
        # self.controls_layout.addWidget(self.coadd_button)


        # Save Spectrum button
        self.save_button = QPushButton('Save Spectrum')
        self.save_button.clicked.connect(self.save_spectrum)
        self.controls_layout.addWidget(self.save_button)

        # Load Spectrum button
        self.load_spectrum_button = QPushButton('Load Spectrum')
        self.load_spectrum_button.clicked.connect(self.load_spectrum)
        self.controls_layout.addWidget(self.load_spectrum_button)

        # Redshift label and input
        self.redshift_label = QLabel('Redshift:')
        self.redshift_label.setWordWrap(True)
        self.controls_layout.addWidget(self.redshift_label)

        self.redshift_input = QLineEdit(self)
        # self.redshift_input.setPlaceholderText("Enter redshift value")
        self.redshift_input.setText(str(self.redshift))
        self.redshift_input.returnPressed.connect(self.update_redshift)
        self.redshift_input.setMinimumWidth(100)
        self.controls_layout.addWidget(self.redshift_input)

        self.scale_label = QLabel('Scale by:')
        self.scale_label.setWordWrap(True)
        self.controls_layout.addWidget(self.scale_label)

        self.scale_input = QLineEdit(self)
        # self.scale_input.setPlaceholderText(str(self.scale_value))
        self.scale_input.setText(f'{self.scale_value:.2e}')
        self.scale_input.returnPressed.connect(self.update_scale)
        self.scale_input.setMinimumWidth(100)
        self.controls_layout.addWidget(self.scale_input)

        # Plot expected line locations button
        self.plot_lines_button = QPushButton('Plot Expected Line Locations')
        self.plot_lines_button.setEnabled(False)
        self.controls_layout.addWidget(self.plot_lines_button)

        # Line selection dropdown
        self.surrounding_label = QLabel('Plot area surrounding:')
        self.surrounding_label.setWordWrap(True)
        self.controls_layout.addWidget(self.surrounding_label)

        self.line_dropdown = QComboBox(self)
        self.line_dropdown.addItems(self.line_labels)
        self.line_dropdown.currentIndexChanged.connect(self.zoom_to_line)
        self.controls_layout.addWidget(self.line_dropdown)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.main_layout.addWidget(self.progress_bar)

        # Status bar
        self.statusBar().showMessage('Ready')

    def create_table(self):
        # Table to display flux and flux error for each line
        self.table = QTableWidget(self)
        self.table.setColumnCount(10)  # Reduced from 10 to 7 columns
        self.table.setHorizontalHeaderLabels([
            'Line', 'Flux', 'Flux Error', 
            'Left Lower', 'Left Upper', 
            'Right Lower', 'Right Upper',
            'Continuum Slope', 'Continuum Intercept',
            'Gaussian Fit File'
        ])
        self.table.setRowCount(len(self.line_labels) - 1)  # Skip 'Full Spectrum'

        # Populate the table
        for i, line_label in enumerate(self.line_labels[1:]):
            self.table.setItem(i, 0, QTableWidgetItem(line_label))
            self.table.setItem(i, 1, QTableWidgetItem(''))
            self.table.setItem(i, 2, QTableWidgetItem(''))
            self.table.setItem(i, 3, QTableWidgetItem(''))  # Left lower bound
            self.table.setItem(i, 4, QTableWidgetItem(''))  # Left upper bound
            self.table.setItem(i, 5, QTableWidgetItem(''))  # Right lower bound
            self.table.setItem(i, 6, QTableWidgetItem(''))  # Right upper bound
            self.table.setItem(i, 7, QTableWidgetItem(''))  # Continuum Slope
            self.table.setItem(i, 8, QTableWidgetItem(''))  # Continuum Intercept
            self.table.setItem(i, 9, QTableWidgetItem(''))  # Gaussian fit file (if applicable)

        self.main_layout.addWidget(self.table)
        self.table.resizeColumnsToContents()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Connect the row selection signal
        self.table.itemSelectionChanged.connect(self.on_table_selection_change)

    def on_table_selection_change(self):
        # Get the selected row(s)
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows:
            row_index = selected_rows[0].row()
            # Get the line label from the first column
            line_item = self.table.item(row_index, 0)
            if line_item:
                line_label = line_item.text()
                # Update the display label
                self.selected_line_display.setText(line_label)
                # Find the matching index in line_labels (add 1 because we skip 'Full Spectrum')
                self.current_line = self.line_labels.index(line_label)
                # Enable the action buttons
                self.action_button.setEnabled(True)
                self.subtract_continuum_button.setEnabled(True)
                self.undo_button.setEnabled(True)

    def update_line_values_from_object_row(self, row_index):
        """Update the line values in the left table from the selected row in the objects table"""
        # Map column indices to line names and their corresponding row in the left table
        line_mappings = {
            "Lya": (1, 0),       # (Column in objects table, Row in left table)
            "Lya_err": (2, 0),
            "OVI": (3, 1),       # OVI corresponds to the second row in the left table
            "OVI_err": (4, 1),
            "CIV": (5, 3),       # CIV corresponds to the fourth row in the left table
            "CIV_err": (6, 3),
            "NV": (7, 2),        # NV corresponds to the third row in the left table
            "NV_err": (8, 2)
        }
        
        # Update each line value
        for line_name, (obj_col, left_row) in line_mappings.items():
            value_item = self.objects_table.item(row_index, obj_col)
            if value_item and value_item.text():
                # Determine if this is a flux or error value
                left_col = 1 if "_err" not in line_name else 2
                # Update the value in the left table
                self.table.setItem(left_row, left_col, QTableWidgetItem(value_item.text()))
    def action_on_selected_line(self):
        # Get the flux method
        flux_method = self.flux_method_dropdown.currentText()
        
        # Depending on the method, call the appropriate function
        if flux_method == "Direct Integration":
            self.find_flux_for_line(self.current_line)
        else:  # "Gaussian Fit"
            # This would be a new method you'd need to implement
            self.fit_gaussian_for_line(self.current_line)

    def subtract_continuum_on_selected_line(self):
        self.subtract_continuum_for_line(self.current_line)

    def undo_continuum_on_selected_line(self):
        self.undo_continuum_for_line(self.current_line)

    def fit_gaussian_for_line(self, line_index):
        # This is a placeholder for the Gaussian fitting functionality
        self.statusBar().showMessage(f"Gaussian fitting not yet implemented for {self.line_labels[line_index]}")
        # jj implement the Gaussian fitting here
    def create_right_panel(self):
        # Create a table for object data
        self.objects_table = QTableWidget()
        self.objects_table.setColumnCount(11)
        self.objects_table.setHorizontalHeaderLabels([
            "Object Name", "Lya", "Lya_err", "OVI", "OVI_err", 
            "CIV", "CIV_err", "NV", "NV_err", "Object_csv", "Object_fits"
        ])
        
        # Make the table take up most of the right panel
        self.right_layout.addWidget(self.objects_table)
        
        # Connect the selection signal
        self.objects_table.itemSelectionChanged.connect(self.on_object_selection_change)
        
        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        
        # Add the "load csv" button
        self.load_csv_button = QPushButton("Load CSV")
        self.load_csv_button.clicked.connect(self.load_objects_csv)
        button_layout.addWidget(self.load_csv_button)
        
        # Add the "save csv" button
        self.save_csv_button = QPushButton("Save CSV")
        self.save_csv_button.clicked.connect(self.save_objects_csv)
        button_layout.addWidget(self.save_csv_button)
        
        # Add the button layout to the right panel
        self.right_layout.addLayout(button_layout)
        
        self.create_detailed_controls()
        
        # Set the stretch factors to make the table take up most of the space
        self.right_layout.setStretch(0, 4)  # Table gets most space
        self.right_layout.setStretch(1, 1)  # Buttons get less space
        self.right_layout.setStretch(2, 2)  # Detailed controls get more space

    def on_object_selection_change(self):
        # Get the selected row
        selected_rows = self.objects_table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_index = selected_rows[0].row()
        
        # Get the csv and fits file paths
        csv_item = self.objects_table.item(row_index, 9)  # Object_csv column
        fits_item = self.objects_table.item(row_index, 10)  # Object_fits column
        
        if not csv_item or not fits_item:
            return
            
        csv_path = csv_item.text()
        fits_path = fits_item.text()
        
        # Check if the paths exist
        if os.path.exists(csv_path) and os.path.exists(fits_path):
            # Load the CSV file (which should contain redshift info and possibly other metadata)
            self.load_csv(csv_path)
            
            # Load the spectrum from the fits file
            self.load_spectrum_from_path(fits_path)
            
            # Update the line values in the table
            self.update_line_values_from_object_row(row_index)

    def load_spectrum_from_path(self, fits_path):
        """Load a spectrum directly from a fits file path"""
        try:
            if fits_path.endswith('.fits'):
                # Load the spectrum from the FITS file
                hdul = fits.open(fits_path)
                data = hdul[1].data
                self.coadded_spectrum = {
                    'wave': data['Wavelength'],
                    'flux': data['Flux']*self.scale_value,
                    'error_up': data['Error_Up']*self.scale_value,
                    'error_down': data['Error_Down']*self.scale_value,
                    'counts': data.get('gcounts', np.zeros_like(data['Wavelength']))  # Fallback if not present
                }
                
                # Try to get redshift from header
                try:
                    self.redshift = hdul[1].header.get('REDSHIFT', self.redshift)
                    self.redshift_input.setText(str(self.redshift))
                except:
                    pass
                    
                hdul.close()
                
                self.plot_spectrum()
                self.statusBar().showMessage(f'Spectrum loaded from {fits_path}')
            elif fits_path.endswith('.npz'):
                # Load the spectrum data from the NPZ file
                data = np.load(fits_path)

                # Restore the co-added spectrum and redshift
                self.coadded_spectrum = {
                    'wave': data['wave'],
                    'flux': data['flux']*self.scale_value,
                    'error_up': data['error_up']*self.scale_value,
                    'error_down': data['error_down']*self.scale_value,
                    'counts': data.get('counts', np.zeros_like(data['wave']))  # Fallback if not present
                }
                self.redshift = data['redshift']
                self.redshift_input.setText(str(self.redshift))

                self.plot_spectrum()
                self.statusBar().showMessage(f'Spectrum loaded from {fits_path}')
            else:
                self.statusBar().showMessage('Unsupported file format.')
        except Exception as e:
            self.statusBar().showMessage(f'Error loading spectrum: {e}')
    def load_csv_with_fits_paths(self):
        options = QFileDialog.Options()
        csv_file, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if csv_file:
            self.load_csv(csv_file)

    def load_objects_csv(self):
        options = QFileDialog.Options()
        csv_file, _ = QFileDialog.getOpenFileName(
            self, "Open Objects CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if csv_file:
            try:
                with open(csv_file, 'r') as file:
                    import csv
                    reader = csv.reader(file)
                    header = next(reader)  # Skip header row
                    
                    # Clear the table
                    self.objects_table.setRowCount(0)
                    
                    # Add rows to the table
                    for row_num, row_data in enumerate(reader):
                        self.objects_table.insertRow(row_num)
                        for col_num, data in enumerate(row_data):
                            item = QTableWidgetItem(data)
                            self.objects_table.setItem(row_num, col_num, item)
                    
                self.statusBar().showMessage(f'Loaded objects CSV: {csv_file}')
            except Exception as e:
                self.statusBar().showMessage(f'Error loading objects CSV: {e}')

    def create_detailed_controls(self):
        # Create a frame with a border for the detailed controls
        self.detailed_controls_frame = QFrame()
        self.detailed_controls_frame.setFrameShape(QFrame.StyledPanel)
        self.detailed_controls_frame.setFrameShadow(QFrame.Raised)
        self.detailed_controls_frame.setMinimumHeight(200)
    
        # Create a layout for the detailed controls
        detailed_layout = QVBoxLayout(self.detailed_controls_frame)
        
        # Add "Selected line:" label and value display
        line_selection_layout = QHBoxLayout()
        line_selection_label = QLabel("Selected line:")
        line_selection_layout.addWidget(line_selection_label)
        
        self.selected_line_display = QLabel("None")
        self.selected_line_display.setStyleSheet("font-weight: bold;")
        line_selection_layout.addWidget(self.selected_line_display)
        line_selection_layout.addStretch()
        
        detailed_layout.addLayout(line_selection_layout)
        
        # Add "Flux method:" label and dropdown
        method_layout = QHBoxLayout()
        method_label = QLabel("Flux method:")
        method_layout.addWidget(method_label)
        
        self.flux_method_dropdown = QComboBox()
        self.flux_method_dropdown.addItems(["Direct Integration", "Gaussian Fit"])
        method_layout.addWidget(self.flux_method_dropdown)
        method_layout.addStretch()
    
        detailed_layout.addLayout(method_layout)
        
        # Add the action buttons in a horizontal layout
        buttons_layout = QHBoxLayout()
        
        self.action_button = QPushButton("Find Flux and Error")
        self.action_button.clicked.connect(self.action_on_selected_line)
        buttons_layout.addWidget(self.action_button)
        
        self.subtract_continuum_button = QPushButton("Subtract Continuum")
        self.subtract_continuum_button.clicked.connect(self.subtract_continuum_on_selected_line)
        buttons_layout.addWidget(self.subtract_continuum_button)
        
        self.undo_button = QPushButton("Undo Continuum")
        self.undo_button.clicked.connect(self.undo_continuum_on_selected_line)
        buttons_layout.addWidget(self.undo_button)
        
        detailed_layout.addLayout(buttons_layout)
        
        # Add spacing at the bottom
        detailed_layout.addStretch()
        
        # Add the frame to the right layout
        self.right_layout.addWidget(self.detailed_controls_frame)
        
        # Initially disable the buttons until a line is selected
        self.action_button.setEnabled(False)
        self.subtract_continuum_button.setEnabled(False)
        self.undo_button.setEnabled(False)
    def save_objects_csv(self):
        options = QFileDialog.Options()
        csv_file, _ = QFileDialog.getSaveFileName(
            self, "Save Objects CSV", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if csv_file:
            try:
                with open(csv_file, 'w', newline='') as file:
                    import csv
                    writer = csv.writer(file)
                    
                    # Write header
                    headers = []
                    for col in range(self.objects_table.columnCount()):
                        headers.append(self.objects_table.horizontalHeaderItem(col).text())
                    writer.writerow(headers)
                    
                    # Write data
                    for row in range(self.objects_table.rowCount()):
                        row_data = []
                        for col in range(self.objects_table.columnCount()):
                            item = self.objects_table.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('')
                        writer.writerow(row_data)
                    
                self.statusBar().showMessage(f'Saved objects CSV: {csv_file}')
            except Exception as e:
                self.statusBar().showMessage(f'Error saving objects CSV: {e}')
    def load_csv(self, csv_file_path):
        try:
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)

                # The first line should contain the redshift
                if lines:
                    self.redshift = float(lines[0][0])
                    self.confirm_redshift()

                # The remaining lines should contain FITS file paths and grating information
                self.fits_file_paths = []
                for row in lines[1:]:
                    print(f"Row: {row}")
                    if len(row) == 2:  # Ensure that each row contains exactly two items
                        self.fits_file_paths.append((row[0], row[1]))
                    else:
                        self.statusBar().showMessage(f'Invalid CSV format on line: {row}')
                        return

            self.statusBar().showMessage(f'Loaded {len(self.fits_file_paths)} FITS file paths from CSV')
            self.load_fits_files()
        except Exception as e:
            self.statusBar().showMessage(f'Error loading CSV: {e}')

    def update_redshift(self):
        try:
            new_redshift = float(self.redshift_input.text())
            self.redshift = new_redshift
            self.statusBar().showMessage(f'Redshift updated to {self.redshift}')
            if self.coadded_spectrum:
                self.plot_spectrum()
        except ValueError:
            self.statusBar().showMessage('Invalid redshift value')
            self.redshift_input.setText(str(self.redshift))

    def update_scale(self):
        try:
            new_scale = float(self.scale_input.text())
            self.scale_value = new_scale
            self.statusBar().showMessage(f'Scale value updated to {self.scale_value:.2e}')
            self.scale_input.setText(f"{self.scale_value:.2e}")  # Update with scientific notation
            if self.coadded_spectrum:
                self.plot_spectrum()
        except ValueError:
            self.statusBar().showMessage('Invalid scale value')
            self.scale_input.setText(f"{self.scale_value:.2e}")  # Revert with scientific notation

    def save_spectrum(self):
        if self.coadded_spectrum is None:
            self.statusBar().showMessage('No co-added spectrum to save.')
            return

        # Get the save file path
        options = QFileDialog.Options()
        save_file, _ = QFileDialog.getSaveFileName(
            self, "Save Spectrum", "", "FITS Files (*.fits);;NPZ Files (*.npz);;All Files (*)", options=options
        )

        if save_file:
            try:
                if save_file.endswith('.npz'):
                    # Save the co-added spectrum and redshift as an NPZ file
                    np.savez(save_file, wave=self.coadded_spectrum['wave'], 
                            flux=self.coadded_spectrum['flux'],
                            error_up=self.coadded_spectrum['error_up'],
                            error_down=self.coadded_spectrum['error_down'],
                            redshift=self.redshift)
                    self.statusBar().showMessage(f'Spectrum saved to {save_file}')
                elif save_file.endswith('.fits'):
                    # Save the spectrum as a FITS file
                    col1 = fits.Column(name='Wavelength', format='E', array=self.coadded_spectrum['wave'])
                    col2 = fits.Column(name='Flux', format='E', array=self.coadded_spectrum['flux'])
                    col3 = fits.Column(name='Error_Up', format='E', array=self.coadded_spectrum['error_up'])
                    col4 = fits.Column(name='Error_Down', format='E', array=self.coadded_spectrum['error_down'])

                    hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
                    hdu.header['REDSHIFT'] = self.redshift
                    hdu.writeto(save_file, overwrite=True)
                    self.statusBar().showMessage(f'Spectrum saved to {save_file}')
                self.update_objects_table_with_saved_spectrum(save_file)
                
            except Exception as e:
                self.statusBar().showMessage(f'Error saving spectrum: {e}')

    def update_objects_table_with_saved_spectrum(self, fits_path):
        """Update the objects table with the saved spectrum"""
        # Check if there's a selected row
        selected_rows = self.objects_table.selectionModel().selectedRows()
        
        if selected_rows:
            # Update the existing row
            row_index = selected_rows[0].row()
            self.objects_table.setItem(row_index, 10, QTableWidgetItem(fits_path))
            
            # Update the flux values from the left table
            self.update_object_row_from_line_values(row_index)
        else:
            # Add a new row
            self.add_new_object_row(fits_path)

    def update_object_row_from_line_values(self, row_index):
        """Update the flux values in the objects table from the left table"""
        # Map line names to their corresponding column in the objects table and row in the left table
        line_mappings = {
            "Lya": (1, 0),       # (Column in objects table, Row in left table)
            "Lya_err": (2, 0),
            "OVI": (3, 1),
            "OVI_err": (4, 1),
            "CIV": (5, 3),
            "CIV_err": (6, 3),
            "NV": (7, 2),
            "NV_err": (8, 2)
        }
        
        # Update each value
        for line_name, (obj_col, left_row) in line_mappings.items():
            # Determine if this is a flux or error value
            left_col = 1 if "_err" not in line_name else 2
            # Get value from left table
            item = self.table.item(left_row, left_col)
            if item and item.text():
                self.objects_table.setItem(row_index, obj_col, QTableWidgetItem(item.text()))
    # def save_spectrum(self):
    #     if self.coadded_spectrum is None:
    #         self.statusBar().showMessage('No co-added spectrum to save.')
    #         return

    #     # Get the save file path
    #     options = QFileDialog.Options()
    #     save_file, _ = QFileDialog.getSaveFileName(
    #         self, "Save Spectrum", "", "FITS Files (*.fits);;NPZ Files (*.npz);;All Files (*)", options=options
    #     )

    #     if save_file:
    #         try:
    #             if save_file.endswith('.npz'):
    #                 # Save the co-added spectrum and redshift as an NPZ file
    #                 np.savez(save_file, wave=self.coadded_spectrum['wave'], 
    #                         flux=self.coadded_spectrum['flux'],
    #                         error_up=self.coadded_spectrum['error_up'],
    #                         error_down=self.coadded_spectrum['error_down'],
    #                         redshift=self.redshift)
    #                 self.statusBar().showMessage(f'Spectrum saved to {save_file}')
    #             elif save_file.endswith('.fits'):
    #                 # Save the spectrum as a FITS file
    #                 col1 = fits.Column(name='Wavelength', format='E', array=self.coadded_spectrum['wave'])
    #                 col2 = fits.Column(name='Flux', format='E', array=self.coadded_spectrum['flux'])
    #                 col3 = fits.Column(name='Error_Up', format='E', array=self.coadded_spectrum['error_up'])
    #                 col4 = fits.Column(name='Error_Down', format='E', array=self.coadded_spectrum['error_down'])

    #                 hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
    #                 hdu.header['REDSHIFT'] = self.redshift
    #                 hdu.writeto(save_file, overwrite=True)
    #                 self.statusBar().showMessage(f'Spectrum saved to {save_file}')
    #             else:
    #                 self.statusBar().showMessage('Unsupported file format.')

    #         except Exception as e:
    #             self.statusBar().showMessage(f'Error saving spectrum: {e}')


    def load_spectrum(self):
        # Get the load file path
        options = QFileDialog.Options()
        load_file, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum", "", "FITS Files (*.fits);;NPZ Files (*.npz);;All Files (*)", options=options
        )

        if load_file:
            try:
                if load_file.endswith('.npz'):
                    # Load the spectrum data from the NPZ file
                    data = np.load(load_file)

                    # Restore the co-added spectrum and redshift
                    self.coadded_spectrum = {
                        'wave': data['wave'],
                        'flux': data['flux']*self.scale_value,
                        'error_up': data['error_up']*self.scale_value,
                        'error_down': data['error_down']*self.scale_value
                    }
                    self.redshift = data['redshift']
                    self.redshift_input.setText(str(self.redshift))
                    

                    self.plot_spectrum()
                    self.statusBar().showMessage(f'Spectrum loaded from {load_file}')
                elif load_file.endswith('.fits'):
                    # Load the spectrum from the FITS file
                    hdul = fits.open(load_file)
                    data = hdul[1].data
                    self.coadded_spectrum = {
                        'wave': data['Wavelength'],
                        'flux': data['Flux']*self.scale_value,
                        'error_up': data['Error_Up']*self.scale_value,
                        'error_down': data['Error_Down']*self.scale_value,
                        'counts': data['gcounts']
                    }
                    self.redshift = hdul[1].header['REDSHIFT']
                    self.redshift_input.setText(str(self.redshift))

                    self.plot_spectrum()
                    self.statusBar().showMessage(f'Spectrum loaded from {load_file}')
                else:
                    self.statusBar().showMessage('Unsupported file format.')
            except Exception as e:
                self.statusBar().showMessage(f'Error loading spectrum: {e}')


    def confirm_redshift(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Redshift Confirmation')
        msg_box.setText(f'Is this your redshift? {self.redshift}')
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec_()

        if result == QMessageBox.No:
            redshift, ok = QInputDialog.getDouble(
                self, "Enter Redshift", "Redshift:", self.redshift, -10.0, 10.0, 3
            )
            if ok:
                self.redshift = redshift
                self.redshift_input.setText(str(self.redshift))
                if self.coadded_spectrum:
                    self.plot_spectrum()
        else:
            self.redshift_input.setText(str(self.redshift))

    def load_fits_files(self):
        self.spectrum_data = [self.load_and_prepare_fits(file_info) for file_info in self.fits_file_paths]
        if self.spectrum_data:
            self.plot_lines_button.setEnabled(True)

    def load_and_prepare_fits(self, file_info):
        try:
            file_name, grating = file_info  # Unpack file path and grating from the tuple
            self.grating_data.append(grating)
            if grating == 'G140L':
                # Use the standard prepare function for G140L grating
                return prepare(file_name)
            else:
                # Use prepare_other_grating for non-G140L gratings
                # try:
                if grating not in self.grating_data:
                    self.grating_data.append(grating)
                return prepare_other_grating(file_name, grating=grating)
                # except Exception as e:
                #     self.statusBar().showMessage(f'Error prepping non-G140L FITS file {file_name}: {e}')
                #     print((f'Error prepping non-G140L FITS file {file_name}: {e}'))
        except Exception as e:
            self.statusBar().showMessage(f'Error loading FITS file {file_name}: {e}')
            return None

    def subtract_continuum_for_line(self, line_index):
        """Subtract the selected continuum for the given line and update the plot."""
        # Retrieve the continuum bounds from the table
        table_row = line_index - 1
        left_lower = self.table.item(table_row, 3).text()
        left_upper = self.table.item(table_row, 4).text()
        right_lower = self.table.item(table_row, 5).text()
        right_upper = self.table.item(table_row, 6).text()
        slope = self.table.item(table_row, 7).text()
        intercept = self.table.item(table_row, 8).text()

        if not (left_lower and left_upper and right_lower and right_upper):
            self.statusBar().showMessage('Continuum bounds are not set for this line.')
            return

        try:
            slope = float(slope)
            intercept = float(intercept)

            # Subtract the continuum using the bounds from the table and plot the new spectrum
            self.apply_continuum_subtraction(slope, intercept)
            self.statusBar().showMessage(f'Continuum subtracted for line: {self.line_labels[line_index]}')
        except ValueError:
            self.statusBar().showMessage('Invalid continuum bounds input.')


    def apply_continuum_subtraction(self, slope, intercept):
        """Perform the continuum subtraction and update the plot."""
        if self.coadded_spectrum is None:
            self.statusBar().showMessage('No co-added spectrum to subtract continuum.')
            return

        wave = self.coadded_spectrum['wave']
        flux = self.coadded_spectrum['flux']

        # # Create masks for the continuum regions
        # left_mask = (wave >= left_lower) & (wave <= left_upper)
        # right_mask = (wave >= right_lower) & (wave <= right_upper)

        # # Calculate the continuum levels
        # left_flux = np.mean(flux[left_mask])
        # right_flux = np.mean(flux[right_mask])

        # # Linear continuum between the two regions
        # continuum = np.interp(wave, [left_lower, right_upper], [left_flux, right_flux])
        continuum = slope * wave + intercept
        # Subtract the continuum from the flux
        self.flux_after_subtraction = flux - continuum

        # Replot the spectrum with the continuum subtracted
        # self.figure.clear()
        # ax = self.figure.add_subplot(111)
        self.ax.plot(wave, self.flux_after_subtraction, label='Spectrum with Continuum Subtracted', color='blue')
        self.ax.set_xlabel('Wavelength')
        self.ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value})')
        self.canvas.draw()

    def undo_continuum_for_line(self, line_index):
        """Undo the continuum subtraction and restore the original spectrum."""
        if self.coadded_spectrum is None:
            self.statusBar().showMessage('No co-added spectrum to undo.')
            return

        # wave = self.coadded_spectrum['wave']
        # flux = self.coadded_spectrum['flux']

        # # Restore the original spectrum
        # self.figure.clear()
        # ax = self.figure.add_subplot(111)
        # ax.plot(wave, flux, label='Original Spectrum', color='black')
        # ax.set_xlabel('Wavelength')
        # ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value:0.2e})')
        # self.canvas.draw()

        #self.statusBar().showMessage(f'Continuum subtraction undone for line: {self.line_labels[line_index]}')
        self.statusBar().showMessage(f'This is buggy... cut for now. Sorry!!')


    def on_coadd_complete(self, coadded_spectrum):
        if coadded_spectrum is None:
            self.statusBar().showMessage('Co-addition failed')
            self.progress_bar.setValue(0)
            return

        self.coadded_spectrum = coadded_spectrum
        self.plot_spectrum()
        self.statusBar().showMessage('Co-addition and plotting complete')
        self.progress_bar.setValue(0)

    def plot_spectrum(self):
        if self.coadded_spectrum:
            wave = self.coadded_spectrum['wave']
            flux = self.coadded_spectrum['flux']
            error_up = self.coadded_spectrum['error_up']
            error_down = self.coadded_spectrum['error_down']

            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            ax = self.ax
            ax.plot(wave, flux, label='Co-added Spectrum', color='black', drawstyle='steps-mid')
            ax.plot(wave, (error_up + error_down)/2, label='Error', color='grey', drawstyle='steps-mid')
            ax.set_xlabel('Wavelength')
            ax.set_xlim(1100, 1880)
            # print(np.max(flux[~np.isnan(flux)]))
            # ax.set_ylim(-0.5, np.max(flux[~np.isnan(flux)]))
            ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value:0.2e})')
            self.plot_expected_lines(ax)
            self.canvas.draw()
        else:
            self.statusBar().showMessage('No co-added spectrum to plot')

    def plot_expected_lines(self, ax=None):
        try:
            redshift = float(self.redshift_input.text()) if self.redshift_input.text() else self.redshift
            observed_lines = self.line_wavelengths * (1 + redshift)
            colors = ['r', 'g', 'b', 'orange', 'purple', 'pink', 'yellow', 'darkblue', 'black']

            if ax is None:
                ax = self.figure.gca()

            for i, line in enumerate(observed_lines[1:]):  # Skip 'Full Spectrum' placeholder
                ax.axvline(x=line, linestyle='--', color=colors[i % len(colors)], label=f'{self.line_labels[i+1]}')
                ax.text(line, ax.get_ylim()[1] * 0.5, self.line_labels[i+1],
                        color=colors[i % len(colors)], rotation=90, verticalalignment='bottom')
        except ValueError:
            self.statusBar().showMessage('Invalid redshift input. Please enter a valid number.')

    def zoom_to_line(self):
        if self.coadded_spectrum:
            selected_line = self.line_dropdown.currentIndex()
            rest_frame_wavelength = self.line_wavelengths[selected_line] * (1 + self.redshift)
            wave = self.coadded_spectrum['wave']
            flux = self.coadded_spectrum['flux']
            error_up = self.coadded_spectrum['error_up']
            error_down = self.coadded_spectrum['error_down']

            if self.line_labels[selected_line] == 'Full Spectrum':
                # Plot the full spectrum
                # self.figure.clear()
                # self.ax = self.figure.add_subplot(111)
                ax = self.ax
                ax.plot(wave, flux, label='Co-added Spectrum', color='black', drawstyle='steps-mid')
                ax.plot(wave, (error_up + error_down)/2, color='grey', drawstyle='steps-mid')
                ax.set_xlabel('Wavelength')
                ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value:0.2e})')
                self.plot_expected_lines(ax)
                self.canvas.draw()
            else:
                zoom_range = (rest_frame_wavelength - 50, rest_frame_wavelength + 50)
                mask = (wave >= zoom_range[0]) & (wave <= zoom_range[1])
                wave_zoom = wave[mask]
                flux_zoom = flux[mask]

                # self.figure.clear()
                # self.ax = self.figure.add_subplot(111)
                ax = self.ax
                ax.plot(wave, flux, label='Co-added Spectrum', color='black', drawstyle='steps-mid')
                ax.plot(wave, (error_up + error_down)/2, color='grey', drawstyle='steps-mid')
                ax.set_xlim(zoom_range)
                ax.set_ylim(0, np.max(flux_zoom[~np.isnan(flux_zoom)]) * 1.1)
                ax.set_xlabel('Wavelength')
                ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value:0.2e})')
                self.plot_expected_lines(ax)
                self.canvas.draw()
        else:
            self.statusBar().showMessage('No co-added spectrum to zoom into')

    def find_flux_for_line(self, line_index):
        self.current_line = line_index
        self.selection_step = 0
        self.statusBar().showMessage(f'Select left continuum start for {self.line_labels[line_index]}')

        # Clear previous selections from the plot
        self.clear_selection_overlays()

        # Connect event handler
        if self.cid is None:
            self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)

    def clear_selection_overlays(self):
        # Remove existing lines and patches
        for line in self.left_continuum_lines + self.right_continuum_lines + self.integration_lines:
            line.remove()
        self.left_continuum_lines = []
        self.right_continuum_lines = []
        self.integration_lines = []

        if self.left_continuum_patch:
            self.left_continuum_patch.remove()
            self.left_continuum_patch = None
        if self.right_continuum_patch:
            self.right_continuum_patch.remove()
            self.right_continuum_patch = None
        if self.integration_patch:
            self.integration_patch.remove()
            self.integration_patch = None

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes is not None:
            x = event.xdata
            if self.selection_step == 0:
                self.left_continuum_start = x
                self.statusBar().showMessage(f'Select left continuum end for {self.line_labels[self.current_line]}')
            elif self.selection_step == 1:
                self.left_continuum_end = x
                self.plot_continuum_region('left')
                self.statusBar().showMessage(f'Select right continuum start for {self.line_labels[self.current_line]}')
            elif self.selection_step == 2:
                self.right_continuum_start = x
                self.statusBar().showMessage(f'Select right continuum end for {self.line_labels[self.current_line]}')
            elif self.selection_step == 3:
                self.right_continuum_end = x
                self.plot_continuum_region('right')
                self.statusBar().showMessage(f'Select integration start for {self.line_labels[self.current_line]}')
            elif self.selection_step == 4:
                self.integration_start = x
                self.statusBar().showMessage(f'Select integration end for {self.line_labels[self.current_line]}')
            elif self.selection_step == 5:
                self.integration_end = x
                self.plot_integration_region()
                self.canvas.mpl_disconnect(self.cid)
                self.cid = None
                self.calculate_flux_for_line()
                return
            self.selection_step += 1

    def plot_continuum_region(self, side):
        if side == 'left':
            start = self.left_continuum_start
            end = self.left_continuum_end
            color = 'blue'
            lines_attr = 'left_continuum_lines'
            patch_attr = 'left_continuum_patch'
        elif side == 'right':
            start = self.right_continuum_start
            end = self.right_continuum_end
            color = 'green'
            lines_attr = 'right_continuum_lines'
            patch_attr = 'right_continuum_patch'

        # Remove previous lines and patches if they exist
        for line in getattr(self, lines_attr):
            line.remove()
        setattr(self, lines_attr, [])

        if getattr(self, patch_attr):
            getattr(self, patch_attr).remove()
            setattr(self, patch_attr, None)

        # Plot vertical lines
        line1 = self.ax.axvline(x=start, color=color, alpha=0.5)
        line2 = self.ax.axvline(x=end, color=color,alpha = 0.5)
        setattr(self, lines_attr, [line1, line2])

        # Shade region
        setattr(self, patch_attr, self.ax.axvspan(start, end, color=color, alpha=0.3))
        self.canvas.draw()

    def plot_integration_region(self):
        # Remove previous lines and patches if they exist
        for line in self.integration_lines:
            line.remove()
        self.integration_lines = []

        if self.integration_patch:
            self.integration_patch.remove()
            self.integration_patch = None

        # Plot vertical lines
        line1 = self.ax.axvline(x=self.integration_start, color='red',alpha=0.5)
        line2 = self.ax.axvline(x=self.integration_end, color='red', alpha=0.5)
        self.integration_lines = [line1, line2]

        # Shade region
        self.integration_patch = self.ax.axvspan(self.integration_start, self.integration_end, color='red', alpha=0.3)
        self.canvas.draw()

    def calculate_flux_for_line(self):
        try:
            table_row = self.current_line - 1
            
            # Get the flux and error values
            print("About to calculate flux...")
            flux_value, flux_error, cont_slope, cont_intercept = self.get_flux(
                self.integration_start, self.integration_end,
                self.left_continuum_start, self.left_continuum_end,
                self.right_continuum_start, self.right_continuum_end
            )
            print(f"Flux calculation returned: {flux_value}, {flux_error}")

            if flux_value is None:
                self.statusBar().showMessage(f'Flux calculation failed for {self.line_labels[self.current_line]}')
                return

            # Update the table with calculated values
            flux_item = QTableWidgetItem(f'{flux_value:.2e}')
            self.table.setItem(table_row, 1, flux_item)
            flux_error_item = QTableWidgetItem(f'{flux_error:.2e}')
            self.table.setItem(table_row, 2, flux_error_item)

            # Update the table with continuum bounds
            self.table.setItem(table_row, 3, QTableWidgetItem(f'{self.left_continuum_start:.2f}'))
            self.table.setItem(table_row, 4, QTableWidgetItem(f'{self.left_continuum_end:.2f}'))
            self.table.setItem(table_row, 5, QTableWidgetItem(f'{self.right_continuum_start:.2f}'))
            self.table.setItem(table_row, 6, QTableWidgetItem(f'{self.right_continuum_end:.2f}'))
            self.table.setItem(table_row, 7, QTableWidgetItem(f'{cont_slope:.2e}'))
            self.table.setItem(table_row, 8, QTableWidgetItem(f'{cont_intercept:.2e}'))

            self.table.resizeColumnsToContents()

            self.statusBar().showMessage(f'Calculated flux for {self.line_labels[self.current_line]}')
        except Exception as e:
            import traceback
            print(f"Error in calculate_flux_for_line: {str(e)}")
            print(traceback.format_exc())
            self.statusBar().showMessage(f'Error calculating flux: {str(e)}')


    def get_continuum(self, continuum_wave, continuum_flux, continuum_error, weights = None):
        
        print(f"Start of get_continuum")
        
        wavelength_array = continuum_wave

        flux_array = continuum_flux
        print(f'Successfully masked for continuum')
        
        if (weights == None):
            weights = np.nan_to_num(1/(continuum_error))
        
        print(f'Weights worked')
        def linear(x, m, b):
            return x*m + b
        

        linear_model = lmfit.Model(linear)
        print(f'linear model worked')
        params_continuum = lmfit.Parameters()
        params_continuum.add('m', value = (np.nanmean(flux_array)/2))
        params_continuum.add('b', value = 0 )
        nan_fluxes = 0.
        total_fluxes = 0.
        for flux in flux_array:
            total_fluxes += 1
            if np.isnan(flux):
                nan_fluxes += 1
        print(f'total fluxes: {total_fluxes}, nan fluxes: {nan_fluxes}')
        try:
            result = linear_model.fit(flux_array, x=wavelength_array, params=params_continuum, weights = weights)
        except Exception as e:
            print(f'Issue with result... error is: {e}')
            result = None
        print(f'Got result')
        print(result.fit_report())
        values = {}
        values['slope'] = result.best_values['m']
        values['intercept'] = result.best_values['b']
        m = values['slope']
        b = values['intercept'] 

        # print(f'slope is {m}, intercept is {b}')
        return values
    def get_flux(self, integ_start, integ_end, cont1_start, cont1_end, cont2_start, cont2_end):
        """
        Calculates the flux and flux error for a given spectral line.

        Parameters
        ----------
        integ_start : float
            Start of the integration range.
        integ_end : float
            End of the integration range.
        cont1_start : float
            Start of the first continuum region.
        cont1_end : float
            End of the first continuum region.
        cont2_start : float
            Start of the second continuum region.
        cont2_end : float
            End of the second continuum region.

        Returns
        -------
        flux : float
            Calculated flux.
        flux_error : float
            Estimated flux error.
        """
        try:
            wavelength_array = self.coadded_spectrum['wave']
            flux_array = self.coadded_spectrum['flux']
            error_array = (self.coadded_spectrum['error_up'] + self.coadded_spectrum['error_down']) / 2
            counts_array = self.coadded_spectrum['counts']
            print(f'Ranges successfully found')
            # Masks for the integration and continuum ranges
            integ_mask = (wavelength_array >= integ_start) & (wavelength_array <= integ_end)
            cont1_mask = (wavelength_array >= cont1_start) & (wavelength_array <= cont1_end)
            cont2_mask = (wavelength_array >= cont2_start) & (wavelength_array <= cont2_end)

            cont_mask = ((wavelength_array >= cont2_start) & (wavelength_array <= cont2_end) | (wavelength_array >= cont2_start) & (wavelength_array <= cont2_end))
            # Mask the flux and error arrays
            flux_array_continuum = flux_array[cont_mask]
            error_array_continuum = error_array[cont_mask]
            wavelength_array_continuum = wavelength_array[cont_mask]
            print(f'mask done')
            # Check for empty masks
            if not np.any(integ_mask):
                self.statusBar().showMessage('No data in the selected integration range.')
                return None, None
            if not np.any(cont1_mask) or not np.any(cont2_mask):
                self.statusBar().showMessage('No data in one or both of the selected continuum ranges.')
                return None, None

            # Calculate continuum levels
            print(f'Finding continuum...')
            continuum_values = self.get_continuum(wavelength_array_continuum, flux_array_continuum, error_array_continuum)
            def linear(x, m, b):
                return x*m + b
            
            slope = continuum_values['slope']
            intercept = continuum_values['intercept']
            print(f'slope is {slope:0.3f}, intercept is {intercept:0.3f}')
            continuum = linear(wavelength_array, slope, intercept)

            # Subtract continuum from flux
            flux_corrected = flux_array[integ_mask] - continuum[integ_mask]
            flux = simpson(flux_corrected, wavelength_array[integ_mask])
            print(f"Calculated flux: {flux}, scaled by {self.scale_value}")
            # Estimate flux error
            counts_array = counts_array[integ_mask]
            total_counts = np.sum(counts_array)
            print(f'total counts worked')
            poisson_conf = pcf(total_counts, interval = 'frequentist-confidence')
            print(f'pcf worked')
            gcounts_error_up = poisson_conf[1] - total_counts
            gcounts_error_down = total_counts - poisson_conf[0] 
            error_poisson_up = (gcounts_error_up/total_counts)*flux
            error_poisson_down = (gcounts_error_down/total_counts)*flux
            # error_integ = error_array[integ_mask]
            flux_error = np.max([error_poisson_up, error_poisson_down])
            print(f'found flux error in get_flux: {flux_error}')

            return flux, flux_error, slope, intercept
        except Exception as e:
            self.statusBar().showMessage(f'Error calculating flux: {e}')
            return None, None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = SpectralFluxApp()
    main_win.show()
    sys.exit(app.exec_())

