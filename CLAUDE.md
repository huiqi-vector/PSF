# overview
This project is to make a GUI for quickly measuring the point spread function (PSF) on lateral and axial directions of our two-photon microscope. Usually 0.2 micron beads are used for imaging and the each beads can be used for calculating it PSF. Users also care about the consistency and homogenity of imaging, so will measure multiple beads in one or multiple subregions in the FOV. 

# goals
- extracting imaging info from the meta data. For a Bruker folder, metadata is the .env data. pixel size on X Y and Z axis can be found : PVStateValue key="micronsPerPixel". 
- calculate PSF for the selected beads. need to get the center_x, center_y for calculating lateral PSF. need to extract the xz and yz slice for calculating axial PSF. the output fitting curve shoud be Intensity (y-axis) - Distance in um (x-axis).
- support manual selection or auto selection (specific number) of the targeting beads. For the auto selection, detect the 'clean' beads and return the pixel postion. 'clean' means the beads shold not overlap with each other and the background signal should be low. 

# GUI structure
- allow loading tiff files and Bruker metadata. If metadata is not available, ask uses to enter the pixel size and z step size. Usually it should be a z-stack multi plane tiff file. Show the basic imaging information, including FOV size, z step size, pixel size. 

- the GUI would have mainly left and right panels. The left panel shows the loaded tiff. add sliding bars for adjusting brightness, contrast, swiching stacks. Allow user to zoom in and out by scrolling the mouse up and down or by clicking + / - buttons. Allow draging the view when zoomed in. 

- allow user to manually select one or multiple beads by a single click and pressing control/cmd and click. the clicking pixel should be center_x and center_y. asign beads id. 

- add a button for auto selecting beads. For this function, the user shold enter the number of beads they want to select. The users can draw 1 or multiple rectangles to constrain the auto selection region. asign beads id.

- the selected beads should show a light red circle around it for visualization. 

- add a button to save the current image with selected beads circle.

- the right upper panel is for showing the PSF results for a single bead. it should be a 2-by-3 plots. the first row (3 plots) is showing the xy, xz and yz plane and the second row (3 plots) is showing the fitting curve with scatters and FWHM value correspondingly. add a dropdown list of bead id to allow users select the bead they want to plot here. also show the pixel position of the plotted bead. add a button to allow saving plot. 

- the right lower panel is for showing the averaged PSF on multiple beads. It should be a 1-by-3 plots, showing the fitting curves on xy, xz and yz. the shade showing the standard deviation. the aveaged FWHM and n number should be shown too. add a button for saving the plot. 

# notes
- please create a uv venv for this project. 
- for the saving plot, .tiff, .svg and .jpg should be support. 
- use the data in the data folder for test. 
- save the plots to the results folder. 
