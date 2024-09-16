
import cv2
import rasterio as rio
from skimage import segmentation, color
from skimage import filters, feature, morphology
from skimage.filters import meijering, frangi
from skimage import measure
import skimage
from skimage.filters import meijering, frangi
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import ImageFilter,Image
import sys
import os
from functools import partial
from scipy.interpolate import splprep, splev
from scipy.integrate import simps
from scipy import optimize, ndimage
from scipy.ndimage.filters import uniform_filter
import warnings
import snakes
from osgeo import ogr, osr, gdal

"""# **IMPORT IMAGE**"""
with rio.open(r"path here") as img:  # open raster dataset
    arr= img.read()  # read as numpy array
    coord = img.profile
a,b,xoff,d,e,yoff,f,g,h=coord['transform']

# mask = np.zeros_like(arr[0])
mask[2:17, 5:40] = 1
# mask[0:10, 10:20] = 1
# mask[132:140, 681:696] = True

"""# **PREPROCESSING**"""

k=arr[2]

''' Application of Gaussian Filter''' 

Gaussian = skimage.filters.gaussian(arr, 6.0)


''' Application of bilateral filtering''' 
# Bilateral = cv2.bilateralFilter(Gaussian[0], d=9, sigmaColor=75, sigmaSpace=75)
plt.imshow(Gaussian[3])
plt.show()


'''Transformation to YUV with rgb '''
# yuvimage = cv2.cvtColor(Gaussian[0], cv2.COLOR_BGR2YUV)

"""# **EDGE DETECTION**"""

def snake_energy(flattened_pts, edge_dist, alpha, beta):
    pts = np.reshape(flattened_pts, (int(len(flattened_pts)/2), 2))
    
    # external energy (favors low values of distance image)
    dist_vals = ndimage.interpolation.map_coordinates(edge_dist, [pts[:,0], pts[:,1]], order=1)
    edge_energy = np.sum(dist_vals)
    external_energy = edge_energy

    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    displacements = pts - prev_pts
    point_distances = np.sqrt(displacements[:,0]**2 + displacements[:,1]**2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist)**2)

    # curvature energy (favors smooth curves)
    curvature_1d = prev_pts - 2*pts + next_pts
    curvature = (curvature_1d[:,0]**2 + curvature_1d[:,1]**2)
    curvature_energy = np.sum(curvature)
    
    return external_energy + alpha*spacing_energy + beta*curvature_energy

    
def fit_snake(coords,pts, edge_dist, alpha=0.5, beta=0.25, nits=80, point_plot=None):

    if point_plot:
        def callback_function(new_pts):
            callback_function.nits += 1
            y = new_pts[0::2]
            x = new_pts[1::2]
            point_plot.set_data(x,y)
            plt.title('%i iterations' % callback_function.nits)
            point_plot.figure.canvas.draw()
            plt.pause(0.02)
        callback_function.nits = 0
    else:
        callback_function = None
    
    # optimize
    cost_function = partial(snake_energy, alpha=alpha, beta=beta, edge_dist=edge_dist)
    options = {'disp':False}
    options['maxiter'] = nits  # FIXME: check convergence
    method = 'BFGS'  # 'BFGS', 'CG', or 'Powell'. 'Nelder-Mead' has very slow convergence
    res = optimize.minimize(cost_function, pts.ravel(), method=method, options=options, callback=callback_function)
    optimal_pts = np.reshape(res.x, (int(len(res.x)/2), 2))
    coords.append(optimal_pts)
    return optimal_pts,coords

# warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
coords=[]

def enhance_ridges(frame, mask=None):
    """Detect ridges (larger hessian eigenvalue)"""
    blurred = filters.gaussian(frame, 2)
    Hxx, Hxy, Hyy = feature.hessian_matrix(blurred, sigma=4.5, mode='nearest', order="xy")
    ridges = feature.hessian_matrix_eigvals((Hxx, Hxy, Hyy))[1]
    
    return np.abs(ridges)



def mask_to_boundary_pts(mask, pt_spacing=10):

    # interpolate boundary
    boundary_pts = measure.find_contours(mask, 0)[0]
    tck, u = splprep(boundary_pts.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    # get equi-spaced points along spline-interpolated boundary
    x_diff, y_diff = np.diff(x_new), np.diff(y_new)
    S = simps(np.sqrt(x_diff**2 + y_diff**2))
    N = int(round(S/pt_spacing))

    u_equidist = np.linspace(0, 1, N+1)
    x_equidist, y_equidist = splev(u_equidist, tck, der=0)
    return np.array(list(zip(x_equidist, y_equidist)))


# get boundary points of mask
boundary_pts = mask_to_boundary_pts(mask, pt_spacing=3)
x, y = boundary_pts[:,1], boundary_pts[:,0]


# distance from ridge midlines
ridges = enhance_ridges(k)
thresh = filters.threshold_otsu(ridges)
prominent_ridges = ridges > 0.8*thresh
skeleton = morphology.skeletonize(prominent_ridges)
edge_dist = ndimage.distance_transform_edt(~skeleton)
edge_dist = filters.gaussian(edge_dist, sigma=2)


# distance from skeleton branch points
blurred_skeleton = uniform_filter(skeleton.astype(float), size=3)
corner_im = blurred_skeleton > 4./9
corners_labels = measure.label(corner_im)
corners = np.array([region.centroid for region in measure.regionprops(corners_labels)])



# show the intermediate images
plt.gray()
plt.ion()
plt.subplot(221)
plt.imshow(mask)
plt.title('original image')
plt.axis('off')
plt.subplot(222)
plt.imshow(ridges)
plt.title('ridge filter')
plt.axis('off')
plt.subplot(223)
plt.imshow(skeleton)
plt.plot(corners[:,1], corners[:,0], 'ro')
plt.title('ridge skeleton w/ branch points')
plt.axis('off')
plt.subplot(223)
plt.imshow(skeleton)
plt.autoscale(False)
plt.plot(corners[:,1], corners[:,0], 'ro')
plt.title('ridge skeleton w/ branch points')
plt.subplot(224)
plt.imshow(edge_dist)
plt.title('distance transform of skeleton')
plt.axis('off')
plt.ioff()
plt.show()


# show an animation of the fitting procedure
fig = plt.figure()
plt.imshow(k, cmap='gray')
plt.plot(x, y, 'bo')
line_obj, = plt.plot(x, y, 'ro')
plt.axis('off')
    
plt.ion()
plt.pause(0.01)
snake_pts,coords = fit_snake(coords,boundary_pts, edge_dist, nits=10, alpha=0.5, beta=0.2, point_plot=line_obj)
plt.ioff()
plt.pause(0.01)
plt.show()
print(coords)

"""# **KML FILE GENERATION**"""

ds = gdal.Open(r"path to tif image")
def pixel2coord(x, y):
    """Returns global coordinates from coordinates x,y of the pixel"""
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff
    return(xp, yp)

# get the existing coordinate system
old_cs= osr.SpatialReference()
old_cs.ImportFromWkt(ds.GetProjectionRef())

# create the new coordinate system
wgs84_wkt = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
new_cs = osr.SpatialReference()
new_cs.ImportFromWkt(wgs84_wkt)

# create a transform object to convert between coordinate systems
transform = osr.CoordinateTransformation(old_cs,new_cs)

finalcood=[]
for i in range(len(coords[0])):
  row, col=coords[0][i][0], coords[0][i][1]
  x,y = pixel2coord(col,row)
  lonx, latx, z = transform.TransformPoint(x,y)
  newcood=[]
  newcood.append(latx)
  newcood.append(lonx)
  finalcood.append(newcood)
print(finalcood)

import os

filename = input("Enter filename: ")
filepath ="path to save file"

if not os.path.exists(filepath):
    os.makedirs(filepath)
filepath = os.path.abspath(filepath)
full_filename = os.path.join(filepath, filename + ".kml")
content='''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>
	<name>somaiya karnataka</name>
	<gx:CascadingStyle kml:id="__managed_style_0D25FC56F627D6502F92">
		<Style>
			<IconStyle>
				<scale>1.2</scale>
				<Icon>
					<href>https://earth.google.com/earth/rpc/cc/icon?color=1976d2&amp;id=2000&amp;scale=4</href>
				</Icon>
				<hotSpot x="64" y="128" xunits="pixels" yunits="insetPixels"/>
			</IconStyle>
			<LabelStyle>
			</LabelStyle>
			<LineStyle>
				<color>ff2dc0fb</color>
				<width>4.8</width>
			</LineStyle>
			<PolyStyle>
				<color>40ffffff</color>
			</PolyStyle>
			<BalloonStyle>
				<displayMode>hide</displayMode>
			</BalloonStyle>
		</Style>
	</gx:CascadingStyle>
	<gx:CascadingStyle kml:id="__managed_style_1BDFCD807D27D6502F91">
		<Style>
			<IconStyle>
				<Icon>
					<href>https://earth.google.com/earth/rpc/cc/icon?color=1976d2&amp;id=2000&amp;scale=4</href>
				</Icon>
				<hotSpot x="64" y="128" xunits="pixels" yunits="insetPixels"/>
			</IconStyle>
			<LabelStyle>
			</LabelStyle>
			<LineStyle>
				<color>ff2dc0fb</color>
				<width>3.2</width>
			</LineStyle>
			<PolyStyle>
				<color>40ffffff</color>
			</PolyStyle>
			<BalloonStyle>
				<displayMode>hide</displayMode>
			</BalloonStyle>
		</Style>
	</gx:CascadingStyle>
	<StyleMap id="__managed_style_0D33FDA43627D6502F91">
		<Pair>
			<key>normal</key>
			<styleUrl>#__managed_style_1BDFCD807D27D6502F91</styleUrl>
		</Pair>
		<Pair>
			<key>highlight</key>
			<styleUrl>#__managed_style_0D25FC56F627D6502F92</styleUrl>
		</Pair>
	</StyleMap>
	<Placemark id="00332148CB27D6502F84">
		<name>Untitled Polygon</name>
		<LookAt>
			<longitude>'''+str(finalcood[0][0])+'''</longitude>
			<latitude>'''+str(finalcood[0][1])+'''</latitude>
			<altitude>574.7972023444377</altitude>
			<heading>0</heading>
			<tilt>0</tilt>
			<gx:fovy>35</gx:fovy>
			<range>256.9985061937769</range>
			<altitudeMode>absolute</altitudeMode>
		</LookAt>
		<styleUrl>#__managed_style_0D33FDA43627D6502F91</styleUrl>
		<Polygon>
			<outerBoundaryIs>
				<LinearRing>
					<coordinates>'''
kmlcoord='''						'''
for i in range(len(finalcood)):
  kmlcoord+=str(finalcood[i][0])+","+str(finalcood[i][1])+","+"574.4075346802837 " 
content2='''					</coordinates>
				</LinearRing>
			</outerBoundaryIs>
		</Polygon>
	</Placemark>
</Document>
</kml>'''
with open(full_filename, "w") as f:
    f.write(content+kmlcoord+content2)

print("File created:", full_filename)