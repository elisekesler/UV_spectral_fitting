import sys
import csv
from functools import partial
import numpy as np
from scipy.integrate import simpson
from astropy.table import vstack
from astropy.io import fits
import lmfit
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QHBoxLayout, QLineEdit, QProgressBar,
    QInputDialog, QMessageBox, QComboBox, QTableWidget,
    QTableWidgetItem
)
from astropy.stats import poisson_conf_interval as pcf
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from prepare_spec import coadd, prepare, prepare_other_grating, combine_tables
import pickle

#try pyqt for getting a separate plot window when zooming in or doing continuum
#matplotlib.lines.Line2D can be saved as an object and later modified, could be good for replotting
class CoaddWorker(QThread):
    # This is here so it's possible to have a progress bar appear during coadding 
    # Might be redundant code
    update_progress = pyqtSignal(int)
    coadd_complete = pyqtSignal(object)

    def __init__(self, spectrum_data, grating_data, G140L_range=None, G130M_range=None, parent=None):
        super().__init__(parent)
        self.spectrum_data = spectrum_data
        self.grating_data = grating_data
        self.G140L_range = G140L_range  
        self.G130M_range = G130M_range  
        self.parent = parent  

    def run(self):
        try:
            G140L_list = [data for data, grating in zip(self.spectrum_data, self.grating_data) if grating == 'G140L']
            G130M_list = [data for data, grating in zip(self.spectrum_data, self.grating_data) if grating == 'G130M']

            if G140L_list and G130M_list and self.G140L_range and self.G130M_range:
                G140L_table = vstack(G140L_list)
                G130M_table = vstack(G130M_list)

                # Combine tables using the provided ranges
                combined_table = combine_tables(
                    G140L_table, G130M_table, 
                    self.G140L_range[0], self.G130M_range[0], 
                    self.G130M_range[0], self.G130M_range[1], 
                    self.G130M_range[1], self.G140L_range[1]
                )
            elif G140L_list:
                combined_table = vstack(G140L_list)
            else:
                combined_table = vstack(self.spectrum_data)

            delta = 1

            # Simulate progress updates
            for i in range(1, 101):
                self.update_progress.emit(i)
                self.msleep(50)  # Simulate some delay for each step -- this is not a real progress bar sadly

            coadded_spectrum = coadd(combined_table, delta)
            self.coadd_complete.emit(coadded_spectrum)
        except Exception as e:
            print(f'Error while coadding: {e}')
            self.statusBar().showMessage(f'Error while coadding: {e}')
            self.coadd_complete.emit(None)



class SpectralFluxApp(QMainWindow):
    # Main app with all the doohickeys and functionalities
    def __init__(self, csv_file_path=None):
        super().__init__()

        self.init_variables()
        self.create_main_widget()
        self.create_plot_area()
        self.create_controls()
        self.create_table()

        self.setWindowTitle('Spectral Flux Measurement')
        self.setGeometry(100, 100, 1000, 800)

    def init_variables(self):
        self.spectrum_data = []
        self.grating_data = []
        self.coadded_spectrum = None
        self.fits_file_paths = []
        self.redshift = 0.0
        self.selection_step = 0
        self.cid = None  # Connection ID for event handler
        self.scale_value = 1e10 #value to scale the flux by

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
        self.main_layout = QVBoxLayout(self.main_widget)

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
        self.redshift_input.setPlaceholderText("Enter redshift value")
        self.controls_layout.addWidget(self.redshift_input)

        self.scale_label = QLabel('Scale by:')
        self.scale_label.setWordWrap(True)
        self.controls_layout.addWidget(self.scale_label)

        self.scale_input = QLineEdit(self)
        self.scale_input.setPlaceholderText(str(self.scale_value))
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
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels(['Line', 'Flux', 'Flux Error', 
                                            'Left Lower', 'Left Upper', 
                                            'Right Lower', 'Right Upper', 
                                            'Action', 'Subtract Continuum', 'Undo'])
        self.table.setRowCount(len(self.line_labels))

        # Populate the table
        for i, line_label in enumerate(self.line_labels[1:]):
            self.table.setItem(i, 0, QTableWidgetItem(line_label))
            self.table.setItem(i, 1, QTableWidgetItem(''))
            self.table.setItem(i, 2, QTableWidgetItem(''))
            self.table.setItem(i, 3, QTableWidgetItem(''))  # Left lower bound
            self.table.setItem(i, 4, QTableWidgetItem(''))  # Left upper bound
            self.table.setItem(i, 5, QTableWidgetItem(''))  # Right lower bound
            self.table.setItem(i, 6, QTableWidgetItem(''))  # Right upper bound

            # "Find Flux and Error" button
            find_flux_error_button = QPushButton('Find Flux and Error')
            find_flux_error_button.clicked.connect(partial(self.find_flux_for_line, i + 1))
            self.table.setCellWidget(i, 7, find_flux_error_button)

            # "Subtract Continuum" button
            subtract_button = QPushButton('Subtract Continuum')
            subtract_button.clicked.connect(partial(self.subtract_continuum_for_line, i + 1))
            self.table.setCellWidget(i, 8, subtract_button)

            # "Undo Continuum" button
            undo_button = QPushButton('Undo Continuum')
            undo_button.clicked.connect(partial(self.undo_continuum_for_line, i + 1))
            self.table.setCellWidget(i, 9, undo_button)

        self.main_layout.addWidget(self.table)
        self.table.resizeColumnsToContents()

    def load_csv_with_fits_paths(self):
        options = QFileDialog.Options()
        csv_file, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if csv_file:
            self.load_csv(csv_file)

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
                else:
                    self.statusBar().showMessage('Unsupported file format.')

            except Exception as e:
                self.statusBar().showMessage(f'Error saving spectrum: {e}')


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

        if not (left_lower and left_upper and right_lower and right_upper):
            self.statusBar().showMessage('Continuum bounds are not set for this line.')
            return

        try:
            left_lower = float(left_lower)
            left_upper = float(left_upper)
            right_lower = float(right_lower)
            right_upper = float(right_upper)

            # Subtract the continuum using the bounds from the table and plot the new spectrum
            self.apply_continuum_subtraction(left_lower, left_upper, right_lower, right_upper)
            self.statusBar().showMessage(f'Continuum subtracted for line: {self.line_labels[line_index]}')
        except ValueError:
            self.statusBar().showMessage('Invalid continuum bounds input.')


    def apply_continuum_subtraction(self, left_lower, left_upper, right_lower, right_upper):
        """Perform the continuum subtraction and update the plot."""
        if self.coadded_spectrum is None:
            self.statusBar().showMessage('No co-added spectrum to subtract continuum.')
            return

        wave = self.coadded_spectrum['wave']
        flux = self.coadded_spectrum['flux']

        # Create masks for the continuum regions
        left_mask = (wave >= left_lower) & (wave <= left_upper)
        right_mask = (wave >= right_lower) & (wave <= right_upper)

        # Calculate the continuum levels
        left_flux = np.mean(flux[left_mask])
        right_flux = np.mean(flux[right_mask])

        # Linear continuum between the two regions
        continuum = np.interp(wave, [left_lower, right_upper], [left_flux, right_flux])

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

        wave = self.coadded_spectrum['wave']
        flux = self.coadded_spectrum['flux']

        # Restore the original spectrum
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(wave, flux, label='Original Spectrum', color='black')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel(f'Flux (erg / s / cm^2 / A ) (scaled by {self.scale_value:0.2e})')
        self.canvas.draw()

        self.statusBar().showMessage(f'Continuum subtraction undone for line: {self.line_labels[line_index]}')


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
            flux_value, flux_error = self.get_flux(
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

            return flux, flux_error
        except Exception as e:
            self.statusBar().showMessage(f'Error calculating flux: {e}')
            return None, None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = SpectralFluxApp()
    main_win.show()
    sys.exit(app.exec_())
