#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Ciaran Robb

utilities module

Description
-----------

Point cloud utils 

"""

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Point_set_processing_3 import *
from CGAL.CGAL_Classification import *
from tqdm import tqdm
from pyntcloud import PyntCloud
import numpy as np
from glob2 import glob
import os
from joblib import Parallel, delayed
import pandas as pd
import geopandas as gpd
import pdal
import json
from osgeo import gdal, ogr, osr
from shapely.wkt import loads
from shapely.geometry import Polygon, LineString
from subprocess import call
from sklearn.model_selection import ParameterGrid
from pointutils.gdal_merge import _merge
import sys
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from skimage.transform import rescale
import skimage.morphology as skm
import dask.array as da
import rasterio
from plantcv import plantcv as pcv
import cv2
import napari
gdal.UseExceptions()
ogr.UseExceptions()


# Notes on pdal pipelines
# To access the internals/outputs
#count = pipeline.execute()
#arrays = pipeline.arrays
#metadata = pipeline.metadata
#log = pipeline.log

def split_cloud(incld, capacity=2000000, writer='ply'):
    
    """
    Split a cloud into smaller tiles of certain capacity
    
    
    Parameters
    ----------
    
    incld: string
             input cloud
             
    outdir: string
              output directory in which tiles will be stored
    
    capacity: int
                size of tile in points
    
    """
    #TODO not working produces only one cloud
    # for ref
    # pdal split --capacity 2000000 --writers.ply.precision=6f BEL06_2_sub4orig.ply out.ply
    # apparently uses chipper to split by capacity what format is output supposed to be?
    # complains if I don't give it a file name but then if I do
    # returns a single cloud. 
    
    hd, tl = os.path.split(incld)
    
    tile = os.path.join(hd, "tile."+writer)
    
    js = [
    incld,
    {
        "type":"filters.chipper",
        "capacity":capacity
    },
    {
        "type":"writers."+writer,
        "filename": tile
    }
    ]
    
    if writer == 'ply':
        js[2]["precision"] = "6f"
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()


def filter_cld(incld, outcld, exp="Classification == 1", writer="las"):
    
    """
    Extract a class from a cloud eg ground or non ground class
    
    
    Parameters
    ----------
    
    incld: string
             input cloud
             
    outcld: string
             output cloud
    
    exp: string
            the filter expression eg "Classification == 1"

    writer: string
            writer type (las, laz, ply, ept)    
    
    """
    js = [
    incld,
    { "type":"filters.expression",
     "expression":exp

    },
    {
        "type":"writers."+writer,
        "filename": outcld
    }
    ]
    
    if writer == 'ply':
        js[2]["precision"] = "6"
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()


def las2ply(incld, outcld):
    
    """
    Translate to ply 
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    
    outcld: string
            output cloud
    
    """    

# TODO pdal screws up RGB values
    js = [incld,

        {
            "type":"writers.ply",
            "filename": outcld,
            "precision": "6f"
        }
        ]

    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()    
    

def colour_cloud(incld, inras, outcld, 
                 scale="Red:1:256, Green:2:256, Blue:3:256",
                 writer='las'):
    
    """
    Colour a cloud with an image
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    inras: string
            input raster
    
    outcld: string
            output cloud
    
    scale: string
             scaling eg where band:scale "Red:1:256, Green:2:256, Blue:3:256"
    
    writer: string
            writer type (las, laz, ply, ept)
    """    
# TODO not working weirdly
#    if incld is None:
#        outcld = incld
        
    js = [incld,
        {
            "type": "filters.colorization",
            "dimensions": scale,
            "raster": inras,
        },
#        {
#            "type": "filters.range",
#            "limits": "Red[1:]"
#        },
        {
            "type":"writers."+writer,
            "filename": outcld}
        ]
    
    if writer == 'ply':
        js[2]["precision"] = "6f"
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()

def clip_cloud_with_cloud(mskcld, incld, outcld, writer='las'):
    
    """
    Clip a pointcloud with the approx boundary of another
    
    Must be same projecttion
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    outcld: string
            output cloud
    
    writer: string
            writer type (las, laz, ply, ept)
    
    """
    poly = cloud_poly(mskcld, outshp=None)
    
    
    js = [
    incld,
    {
        "type":"filters.crop",
        "polygon": poly.wkt
    },
    {
     "type":"writers."+writer,
     "filename":outcld
    }
    ]
    
    if writer == 'ply':
        js[2]["precision"] = "6f"

    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()

    
    

def reproject_cloud(incld, outcld, inproj="ESPG:4326", outproj="ESPG:32630", 
                    reader='las', writer='las'):
    
    """
    Reproject a cloud
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud
    
    inproj: string
            e.g. in form ESPG:4326 or proj4 if that doesn't work
            PROJ 4 RECOMMENDED
    
    outproj: string
            e.g. in form ESPG:4326 or proj4 if that doesn't work
            PROJ 4 RECOMMENDED
            
    reader: string
            reader type (las, laz, ply, ept)
    
    writer: string
            writer type (las, laz, ply, ept)
    """
     
    
    js = [
        {
            "filename": incld,
            "type":"readers."+reader,
            "spatialreference": inproj,
        },
        {
            "type":"filters.reprojection",
            "in_srs": inproj,
            "out_srs": outproj,
        },
        {
            "type":"writers."+writer,
            "filename":outcld
        }
    ]
    if writer == 'ply':
        js[2]["precision"] = "6f"

    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()
    
    
def merge_cloud(inclds, outcld, reader="readers.ply", writer="ply"):
    
    """
    Merge some las/laz files via pdal
    
    Parameters
    ----------
    
    inclds: list or filepath
            list of input clouds or string to dir
            
    outshp: string
            output file (.las/laz)
    
    reader: string
            reader type (las, laz, ply, ept)
    
    writer: string
            writer type (las, laz, ply, ept)
    
    """
    
    # must copy otherwise we end up with a load of pointing
    
    outwrite = "writers."+writer
    
    if type(inclds) is str:
        print('input is a directory')
        # assume a path
        # get the filetype from reader arg
        key = "*" + os.path.splitext(reader)[1] 
        inclds = glob(os.path.join(inclds, key))
        inclds.sort()
    
    js = inclds.copy()
    # it wont accpet the list directly....
    # ...just append the pipeline to the end of it -handy
    
    js.append({"type": "filters.merge"
               })
    #precision is still an issue with some files 
    js.append({
            "type": outwrite,
            "precision": "6f", 
            "filename": outcld
            })
    
    pipeline = pdal.Pipeline(json.dumps(js))
    
    count = pipeline.execute()
    

def cloud_poly(incld, outshp=None, polytype="ESRI Shapefile", edge_sz=1,
               thresh=0, reader="readers.las", spref=None):
    
    """
    Return the non-zero extent as a shapely polygon
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    outshp: string
            output file (optional)
    
    polytype: string
            the ogr term for the output polygon e.g. "ESRI Shapefile" (optional)
    
    edge_sz: int
        min edge size of each line
    
    thresh: int
         threshold
    
    spref: int
            an espg code if the point cloud doesn't have one
    
    Returns
    -------
    
    Shapely polygon
    
    """

#    filters.hexbin.edge_size=10 \
#    --filters.hexbin.threshold=1
    
    js = [incld, 
          {"type" : "filters.hexbin",
           "edge_size": edge_sz,
           "threshold": thresh}]
    
    pipeline = pdal.Pipeline(json.dumps(js))
    
    count = pipeline.execute()
    
    # get the json 
    meta = pipeline.metadata
    metajson = json.loads(meta)
    
    # in json is the polygon required
    boundarywkt = metajson['metadata']['filters.hexbin']['boundary']
    poly = loads(boundarywkt)
    
    if outshp != None:
        # this is a wkt
        proj = osr.SpatialReference()
        
        if spref == None:
            spref = metajson['metadata'][reader]['spatialreference']
            proj.ImportFromWkt(spref)
        else:
            proj.ImportFromEPSG(spref)
        
        out_drv = ogr.GetDriverByName(polytype)
        
        # remove output shapefile if it already exists
        if os.path.exists(outshp):
            out_drv.DeleteDataSource(outshp)
        
        # create the output shapefile
        ootds = out_drv.CreateDataSource(outshp)
        ootlyr = ootds.CreateLayer("extent", proj, geom_type=ogr.wkbPolygon)
        
        # add an ID field
        idField = ogr.FieldDefn("id", ogr.OFTInteger)
        ootlyr.CreateField(idField)
        
        # create the feature and set values
        featureDefn = ootlyr.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        # from shapely wkt export to ogr
        polyogr = ogr.CreateGeometryFromWkt(poly.wkt)
        feature.SetGeometry(polyogr)
        feature.SetField("id", 1)
        ootlyr.CreateFeature(feature)
        feature = None
        
        # Save and close 
        ootds.FlushCache()
        ootds = None
    
    return poly

#TODO
def create_ogr_poly(outfile, spref, file_type="ESRI Shapefile", field="id", 
                     field_dtype=0):
    """
    Create an ogr dataset and layer (convenience)
    
    Parameters
    ----------
    
    outfile: string
                path to ogr file 
    
    spref: wkt or int
        spatial reference either a wkt or espg
    
    file_type: string
                ogr file designation
        
    field: string
            attribute field e.g. "id"
    
    field_type: int or ogr.OFT.....
            ogr dtype of field e.g. 0 == ogr.OFTInteger
        
             
    """   
    proj = osr.SpatialReference()
    #TODO if int assume espg - crude there will be a better way
    if spref is int:
        proj.ImportFromEPSG(spref)
    else:
        proj.ImportFromWkt(spref)
        
    out_drv = ogr.GetDriverByName(file_type)
    
    # remove output shapefile if it already exists
    if os.path.exists(outfile):
        out_drv.DeleteDataSource(outfile)
    
    # create the output shapefile
    ootds = out_drv.CreateDataSource(outfile)
    ootlyr = ootds.CreateLayer("extent", proj, geom_type=ogr.wkbPolygon)
    
    # add the fields
    # ogr.OFTInteger == 0, hence the arg
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    ootlyr.CreateField(idField)
    
    return ootds, ootlyr
    
    

def intersect_clip(incld, inshp, outcld, column, index, writer='las'):
    
    """
    Clip a pointcloud with a polygon if it intersects the polygon
    
    Must be same projecttion
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    inshp: string
            optional output file
            
    outcld: string
            output cloud    
            
    column: string
            the shape column/attribute
    
    index: int/string/float
            the row to filter by
    
    outcld: string
            input cloud

    writer: string
            the writer type eg las, ply
    
    """
    
    cldpoly = cloud_poly(incld)
    pcd
    gdf = gpd.read_file(inshp)
    
    row = gdf[gdf[column]==index]
    
    # if there is more than one, merge them
    if len(row) > 0:
         row = row.dissolve(by=column)
    # for shapely
    wkt = row.geometry.to_wkt()
    inwkt = wkt.iloc[0]
    
    poly = loads(inwkt)
    
    if cldpoly.intersects(poly) == True:
        clip_cloud(incld, inshp, outcld, column, index, writer)
    else:
        print("pointcloud does not intersect polygon")

def atrribute_cloud(incld, inshp, shpcol, cldcol="Classification"):
    
    """
    Add the attribute of a shapefile polygon to the points it contains
    Must be on an existing column.
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    inshp: string
            optional output file
            
    outcld: string
            output cloud    
            
    shpcol: string
            the shape column/attribute
    
    cldcol: string
            the shape column/attribute
    """
    
    # seems very slow....
    js = [
    incld,
    {
        "type":"filtpcders.overlay",
        "dimension": cldcol,
        "datasource": inshp,
#        "layer":"attributes",
        "column": shpcol
    },
    {
        "filename": incld,
#        "scale_x":0.0000001,
#        "scale_y":0.0000001
    }
    ]
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()      
    
def clip_cloud(incld, inshp, outcld, column, index, writer='las'):

    """
    Clip a pointcloud using pdal specifying the attribute and feature number
    Projections MUST be the same for poly & pointcloud
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    inshp: string
            input polygon file

    outcld: string
            output cloud

    column: string
                the shape column/attribute
    
    index: int/string/float
            the row to filter by

    writer: string
            the writer type eg las, ply
    
    """

    gdf = gpd.read_file(inshp)
    
    row = gdf[gdf[column]==index]
    
    # if there is more than one, merge them
    if len(row) > 0:
         row = row.dissolve(by=column)
    
    wkt = row.geometry.to_wkt()
    
    inwkt = wkt.iloc[0]
    # poly1 = loads(inwkt)
    
    # ply precision issue hence we specify here
    write ="writers." + writer
    if writer == 'ply':
        output = {
        "type": write,
        "precision": "6f", 
        "filename":outcld
        }
    else:
         output ={
        "type": write,
        "filename":outcld
        }
                    
    js = {"pipeline": [
    incld,
    {
        "type":"filters.crop",
        "polygon": inwkt
    },
     output

    ]}

    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute() 


# the pdal way on their site....
## seemingly no need to specify esri 
#    js= [
#        bigcld,
#        {
#            "column": "DN",
#            "datasource": maskpoly,
#            "dimension": "Classification",
#            "type": "filters.overlay"
#        },
#        {
#            "limits": "Classification[1:1]",
#            "type": "filters.range"
#        },
#        bigveg
#         ]





def grid_rgb(incld, outfile, fill=True, no_data=0, reader="readers.las", 
               writer="writers.gdal", spref="EPSG:32630", dtype="uint16_t",
               outtype='mean', resolution=0.1, rng_limit=None, para=True):

    
    """
    Grid RGB and intensity into a multiband raster. 
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outfile: string
            output cloud
    
    fill: bool
            whether to fill no data - though could just grid coarser....
    
    no_data: 
            the no_data value to assign to the raster
    
    reader: string
            the pdal reader type (see pdal readers)
    
    writer: string
            the pdal reader type (see pdal writers)
    
    spref: string
            spatial ref in ESPG format
    
    dtype: string
            dtype in pdal format (see https://pdal.io/types.html) 
            e.g. uint16_t, float32
        
    outtype: string
            mean, min, max, idw, count, stdev
            
    resolution: float
            in the unit required
    
    rng_limit: string
        only grid values in a range e.g. "Classification[1:2]" or "label[1:1]"
        as per the pdal convention
    
    para: bool
        whether to process in parallel or not - be careful with big point
        clouds on this front.

    """
    # temporary assumption file type is 3 letters! Change this....
    clrs = ['r', 'g', 'b', 'nir']
    # list of raster to create
    tmplist = [incld[:-4]+c+'.tif' for c in clrs]
    # when pulling from las/ply
    att = ['Red', 'Green', 'Blue', 'Intensity']
    
    # no of threads
    nt = 4
    
    if para == True:
        Parallel(n_jobs=nt, verbose=2)(delayed(grid_cloud)(incld, t, k,
             reader, writer, spref, dtype,
             outtype, resolution, rng_limit) for t, k in zip(tmplist, att))
    else:
        # There could be a minor chance RAM is consumed with big clouds
        # so sequential here in case e.g 170ma points = 70% of 96gb 
        for t, k in zip(tmplist, att):
            grid_cloud(incld, t, k ,reader, writer, spref, dtype,
                 outtype, resolution, rng_limit)
            
    _merge(names=tmplist, out_file=outfile, a_nodata=no_data)
    
    # fill holes - usually required.
    if fill == True:
        print('filling no data holes')
        fill_nodata(outfile, maxSearchDist=5, smoothingIterations=1, 
                    bands=[1, 2, 3, 4])
    
    # dump the temp rasters
    [os.remove(t) for t in tmplist]
    
    

       
    
def grid_cloud(incld, outfile, attribute="label", reader="readers.ply",
               writer="writers.gdal", spref="EPSG:32630", dtype="uint16_t",
               outtype='mean', resolution=0.1, rng_limit=None):
    
    """
    Grid a pointcloud attribute using pdal
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outfile: string
            output cloud
    
    attribute: string
            the pointcloud attribute/dimension to rasterize
            e.g. label, classification etc
    
    reader: string
            the pdal reader type (see pdal readers)
    
    writer: string
            the pdal reader type (see pdal writers)
    
    spref: string
            spatial ref in ESPG format
    
    dtype: string
            dtype in pdal format (see https://pdal.io/types.html) 
            e.g. uint16_t, float32
        
    outtype: string
            mean, min, max, idw, count, stdev
            
    resolution: float
            in the unit required
    
    rng_limit: string
        only grid values in a range e.g. "Classification[1:2]" or "label[1:1]"
        as per the pdal convention

    """
    
    #json from args
    js = [{
                "type": reader,
                "filename":incld,
        	"spatialreference":spref
            },
            {
              "type": writer, 
              "filename": outfile,
              "dimension": attribute,
              "data_type": dtype,
              "output_type": outtype,
              "resolution": resolution
            }]

    if rng_limit != None:
        js.insert(1, {"type":"filters.range", 
                   "limits":rng_limit})
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()
    #log = pipeline.log
    
def grid_cloud_batch(folder, attribute="label", reader="readers.ply", 
                    writer="writers.gdal", spref="EPSG:21818", 
                    dtype="uint16_t", outtype='mean', resolution=0.1, nt=-1,
                    rng_limit=None):
    
    """
    Grid pointclouds attribute using pdal
    
    Parameters
    ----------
    
    folder: string
            input folder containing plys
    
    attribute: string
            the pointcloud attribute/dimension to rasterize
            e.g. label, classification etc
    
    reader: string
            the pdal reader type (see pdal readers)
    
    writer: string
            the pdal reader type (see pdal writers)
    
    spref: string
            spatial ref in ESPG format
    
    dtype: string
            dtype in pdal format (see pdal)
        
    outtype: string
            mean, min or max
            
    resolution: float
            in the unit required

    """
    _ , ext = os.path.splitext(reader)
    
    plylist = glob(os.path.join(folder, '*'+ext))
    plylist.sort()

    outlist = [ply[:-3]+'tif' for ply in plylist]
        
        
    Parallel(n_jobs=nt, verbose=2)(delayed(grid_cloud)(ply, out, attribute,
             reader, writer, spref, dtype,
             outtype, resolution, rng_limit) for ply, out in zip(plylist, outlist))



def pdal_denoise(incld, outcld, method="statistical", multi=3, k=8,
                 remove_outliers=True):
    
    """
    Denoise a pointcloud attribute using pdals outlier filters
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud
            
    method: string
            either statistical or
    
    multi: int
            multiplier
    
    k: int
            k neighbours
    
    remove_outliers: bool
            Whether to remove the outlying points- they are classified as 7
            
    """
    
    js= {
        "pipeline": [
            incld,
            {
                "type": "filters.outlier",
                "method": method,
                "multiplier": multi,
                "mean_k": k
            } ]
        }
    
    if remove_outliers == True:
        pip = {"type": "filters.expression",
               "expression": "Classification == 1"}
        js['pipeline'].append(pip)
    
    # done here as it is erased but rm_out option insertion
    js['pipeline'].append({
            "type": "writers.las",
            "filename": outcld
        })
    

    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()
    


def pdal_thin(incld, outcld, method="filters.sample", radius=5, 
              writer='las'):
    
    """
    Thin a pointcloud attribute using pdal
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud
    
    method: string
            one of  filters.voxelgrid, filters.sample (Poisson)
    
    radius: int
            sample radius
    
    writer: str
           '
            
    """

    js = {"pipeline": [
            incld, {
        "type": method,
        "radius": radius
        },
        {
        "type":"writers."+writer,
        "filename":outcld
        }
        ]}
    
    if writer == 'ply':
        js[2]["precision"] = "6f"
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()

def pdal_smrf(incld, smrf=None, elm=False, scalar=1.25, slope=0.15, threshold=0.5, 
               window=18, clsrange="[1:2]", outcld=None):
    """
    Pdal-based simple morphological filter with optional param sets from the 
    paper
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud (if none), results written to input
    
    elm: bool
        Extended Local Minimum (ELM) method helps to identify low noise points 
        that can adversely affect ground segmentation algorithms
    
    smrf: int 
                From the Pingel (2013) paper tests - helpful perhaps but no
                guarantees of 'good' results!
                Choice of: 
                1 = Mixed vegetation and buildings on hillside,
                2 = Mixed vegetation and buildings,
                3 = Road with bridge,
                4 = Bridge and irregular ground surface,
                5 = Large, irregularly shaped buildings,
                6 = Steep slopes with vegetation,
                7 = Complex building,
                8 = Large gaps in data, irregularly shaped buildings,
                9 = Trains in railway yard,
                10 = Data gaps, vegetation on moderate slopes,
                11 = Steep, terraced slopes1
                12 = Steep, terraced slopes2
                13 = Dense ground cover
                14 = Large gap in data
                15 = Underpass
                    
    scalar: float
            elevation scaling factor (def 1.25)
    
    slope: float
            slope thresh (percent) (def 0.15)
    
    threshold: float
            elevation threshold (def 0.5)

    window: int
            max window size  (def 18)
    """
    
    # the test results from the paper

    if outcld == None:
        outcld = incld
    if smrf != None:
        sms = smrf_params()
        row = sms.iloc[smrf]
        print('Parametrs are:\n', row[2:6])
        params = row[2:6].to_dict()
        params["ignore"]="Classification[7:7]"
        params["type"] ="filters.smrf"

    else:
        params = {
        "type":"filters.smrf",
        "ignore":"Classification[7:7]",
        "scalar": scalar,
        "slope": slope,
        "threshold":threshold,
        "window":window} 
     
    if elm == True:
        js= [incld,
         {"type": "filters.assign",
          "assignment": "Classification[:]=0"},
         {"type": "filters.elm", },
         {"type": "filters.outlier",}]
    else:
        js = [incld]
    
    
    js.append(params)
    js.append({"type":"filters.range",
             "limits":"Classification"+clsrange})
    js.append(outcld)
  
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()

def iter_smrf(incld,  params={'scalar': [1,5,10], 'slope':[0.25,0.5,1], 
                                  'thresh':[0.1,0.5,1], 
                                  'wind':[5,10,20]},
            clsrange="[1:2]", para=False):
    
    """
    Run all possible parameter combinations for the smrf algorithm. 
    Output clouds will be named by param set and same file type
    
    Esoteric task for recent project. See pdal_smrf for an explanation of 
    parameters
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    params: dict
            parameter dict with values to be run
                    
    
    """
    
    # use sklearn to produce the generator
    param_grid = ParameterGrid(params)
    
    # do the task....(could be parallelized...but not sure how much of cgal C++
    # already is)
    for p in tqdm(param_grid):
        
        ootcld =_sm_rename(incld, p)
        pdal_smrf(incld,  scalar=p['scalar'], slope=p['slope'], 
                  threshold=p['thresh'], 
                  window=p['wind'], clsrange=clsrange, outcld=ootcld)

def _sm_rename(incld, p):
    
    """
    For the above smrf iter
    """
    st = str(p)
    st = st.replace("': ", "")
    st = st.replace("{'", "")
    st = st.replace("}", "")
    st = st.replace(", '", "_")
    nm, ext = os.path.splitext(incld)
    ootname = nm+"_"+st+ext

    return ootname

def cgal_normals(incld, outcld=None, k=24, method='jet'):
    
    """
    Normal estimation via cgal
    
    Parameters
    ----------
    
    incld: string
            input cloud

    outcld: string
            output cloud, if none, input will be overwritten
    
    k: int
        k-neighbours
    
    method:
        method of estimating normals one of:
            jet, mst or pca
            
    """
    
    points = Point_set_3(incld)
    
    points.add_normal_map()
    
    if method == 'jet':
        print("Running jet_estimate_normals...")
        jet_estimate_normals(points, k)
        
    elif method == 'mst':
        print("Running mst_orient_normals...")
        mst_orient_normals(points, k)
    
    elif method == 'pca':
        print("Running pca_estimate_normals...")
        pca_estimate_normals(points, k)

    if outcld == None:
        outcld = incld
    
    points.write(outcld)

def cgal_normal_batch(folder, k=24, method='jet', nt=-1):
    
    
    """
    Normal estimation via cgal for ply tiles in a folder/dir
    Normals will be written to inputs
    
    Parameters
    ----------
    
    folder: string
            input directory

    
    k: int
        k-neighbours
    
    method:
        method of estimating normals one of:
            jet, mst or pca
    
    nt: int
        no of threads, default of -1 means all available
            
    """
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()

    Parallel(n_jobs=nt, verbose=2)(delayed(cgal_normals)(ply,
             outcld=None, k=k, method=method)for ply in plylist)

def cgal_smooth(incld, outcld):
    
    """
    Parameters
    ----------
    
    incld: string
         input cloud
         
    outcld: string
            output cloud, if none, input will be overwritten 
    
    """
    
    
    points = Point_set_3(incld)

    k = estimate_global_k_neighbor_scale(points)
    scale = estimate_global_range_scale(points)
    
    jet_smooth_point_set(points, k, neighbor_radius=scale)
    
    if outcld == None:
        outcld = incld
    
    points.write(outcld)

def cgal_edge_upsample(incld, outcld=None, nopoints=1000, sharpness=30.0,
                       edge=1.0, n_radius=-1.0):
    
    """
    Edge aware upsampling
    
    Parameters
    ----------
    
    incld: string
         input cloud
         
    outcld: string
            output cloud, if none, input will be overwritten    
    
    nopoints: int
            no of points retained
    
    sharpness: float
            sharpness angle
    
    edge: float
            edge sensitivity
    
    n_radius: flaat
            neighbor radiu
    """
    
    points = Point_set_3(incld)
    
    edge_aware_upsample_point_set(points, number_of_output_points=nopoints,
                                  sharpness_angle=sharpness,
                                  edge_sensitivity=edge,
                                  neighbor_radius=n_radius)
    
    if outcld == None:
        outcld = incld
    
    points.write(outcld)

def cgal_outlier(incld, outcld=None, k=24, distance=0.1, percentage=100, 
                 recursive=False):
    
    """
    Point cloud outlier removal via cgal
    
    Parameters
    ----------
    
    incld: string
            input cloud
            
    outcld: string
            output cloud, if none, input will be overwritten

    k: int
        k-neighbours
            
    distance: int
            min distance to outliers
            
    percentage: int
            max percentage of points to remove
    
    """
    points = Point_set_3(incld)
    
    if recursive == True:
        nopoints = 1
        while nopoints > 0:
            remove_outliers(points, k, neighbor_radius=0.0,
                        threshold_distance=distance, 
                        threshold_percent=percentage)
            nopoints = points.garbage_size()
            points.collect_garbage()
    else:
        remove_outliers(points, k, neighbor_radius=0.0,
                        threshold_distance=distance)
        print(points.size(), "point(s) remaining,", points.garbage_size(),
              "point(s) removed")
        # bin it
        points.collect_garbage()
    
    if outcld == None:
        outcld = incld
    
    points.write(outcld)

def split_into_classes(incld, field='label', classes=None, folder=None):
    
    """
    Split a classified ply file into seperate classes or output only those 
    selected
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    classes: list of ints
            a list of the classes to create seperate pointclouds with
    
    folder: string 
        the dir (if any) the clouds go into
    
    """
    
    pcd = PyntCloud.from_file(incld)
    
    #recall that pyntcloud child objects are linked, hence the loop
    
    if folder == None:
        folder = os.path.join(os.path.dirname(incld), 'sepCls')
    
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    if classes == None:
        classes = pcd.points[field].unique().tolist()
        classes.sort()
    
    for c in tqdm(classes):
        # must instantiate new one
        newpoints = PyntCloud(pcd.points[pcd.points[field]==c])
        name, ext = os.path.splitext(incld)
        name = os.path.basename(name)
        name = os.path.join(folder, name)
        newpoints.to_file(name+str(c)+ext)

def split_into_classes_batch(folder, field='label', classes=None, nt=-1):
    
    """
    Split a classified ply file into seperate classes or output only those 
    selected
    
    Parameters
    ----------
    
    folder: string
            input cloud
    
    classes: list of ints
            a list of the classes to create seperate pointclouds with
    
    nt: int
        no of threads (def -1 for all available)
    
    
    """
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    

    Parallel(n_jobs=nt, verbose=2)(delayed(split_into_classes)(
            p, field=field, classes=classes) for p in plylist)

    
def cgal_outlier_batch(folder, k=24, distance=0.1, percentage=100, 
                 recursive=False, nt=-1):
    
    """
    Point cloud outlier removal via cgal for multiple files in a dir
    
    Parameters
    ----------
    
    folder: string
            input directory
            
    outcld: string
            output cloud, if none, input will be overwritten

    k: int
        k-neighbours
            
    distance: int
            min distance to outliers
            
    percentage: int
            max percentage of points to remove
    
    nt: int
        no of threads (def -1 for all available)
    
    """
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    outlist = [i[:-4]+'_clean.ply' for i in plylist]
    outlist.sort()
    
    Parallel(n_jobs=nt, verbose=2)(delayed(cgal_outlier)(
            p, o, k, distance, percentage, recursive
            ) for p,o  in zip(plylist, outlist))


def cgal_simplify(incld, outcld=None, method='grid',  k=None):
    
    """
    Simplify a pointcloud via cgal methods
    
    Parameters
    ----------
    
    incld: string
            input cloud

    outcld: string
            output cloud, if none, input will be overwritten
    
    method: string
            a choice of 'grid', 'hierarch', 'wlop'  

    k: int
        k neighbours for spacing, if none it is estimated from data
    
    """
    
    points = Point_set_3(incld)
    
    if k == None:
        k = estimate_global_k_neighbor_scale(points)
        print("K-neighbor scale is", k)
    
    
    avg_space = compute_average_spacing(points, k)
    print("Average point spacing is", avg_space)
    
    print('Using ', method) 
    if method == 'grid':
        grid_simplify_point_set(points, avg_space)
    elif method == 'hierarch':
        hierarchy_simplify_point_set(points)
    elif method == 'wlop':
        wlop = Point_set_3()
        #TODO seem to lose rgb values - perhaps need to get value from nearest
        # point??
        wlop_simplify_and_regularize_point_set(points,  # input
                                       wlop)  # Output
        wlop.write(outcld)
        
    if method == 'grid' or method == 'hierarch':
        print(points.size(), "point(s) remaining,", points.garbage_size(), 
              "point(s) removed")
        
        points.collect_garbage()
        
        points.write(outcld)


def cgal_simplify_batch(folder, method='grid',  k=None, para=True, nt=None):
    
    """
    Simplify  multiple pointclouds via cgal methods - outputs written to same folder

    
    Parameters
    ----------
    
    incld: string
            input folder where pointclouds(ply) reside
    
    method: string
            a choice of 'grid', 'hierarch', 'wlop'  

    k: int
        k neighbours for spacing, if none it is estimated from data

    para: bool
        process in porallel

    nt: int
        no of parallel jobs (if none will be no of input files)   
    """

    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()
    
    outlist = [i[:-4]+method+'.ply' for i in plylist]
    outlist.sort()
    
    if para == False:
        for p, o in tqdm(zip(plylist, outlist)):
            cgal_simplify(p, o, method=method,  k=k)
            
    
    if nt == None:
        nt = len(plylist)
     
    Parallel(n_jobs=nt, verbose=2)(delayed(cgal_simplify)(
            folder, method=method,  k=k) for p in plylist)
    
def del_field(incld, fields):
    
    """
    Delete a ply field

    
    Parameters
    ----------
    
    incld: string
            input folder where pointclouds(ply) reside
    
    fields: list of strings
            the fields to get dumped
    
    """
    pcd = PyntCloud.from_file(incld)
    
    pcd.points = pcd.points.drop(columns=fields)
    
    pcd.to_file(incld)

def write_vrt(infiles, outfile):
    
    """
    Parameters
    ----------
    
    infiles: list of strings
                the input files
    
    outfile: string
                the output .vrt

    """
    
    
    virtpath = outfile
    outvirt = gdal.BuildVRT(virtpath, infiles)
    outvirt.FlushCache()
    outvirt=None 

def resize(inRas, outRas, templateRas):
    
    """
    Resize a raster with the pixel dims of another.
    
    Assumes they overlap exactly (same geo extent)
    
    """
    
    rds = gdal.Open(templateRas)
    xpix = rds.RasterXSize
    ypix = rds.RasterYSize
    
    
    ootRas = gdal.Translate(outRas, inRas, width=xpix, height=ypix)
    ootRas.FlushCache()
    ootRas=None

# for ref....
#   The nump and gdal dtype (ints)
#   {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,"uint32": 4,"int32": 5,
#    "float32": 6, "float64": 7, "complex64": 10, "complex128": 11}

# a numpy gdal conversion dict - this seems a bit long-winded
#        dtypes = {"1": np.uint8, "2": np.uint16,
#              "3": np.int16, "4": np.uint32,"5": np.int32,
#              "6": np.float32,"7": np.float64,"10": np.complex64,
#              "11": np.complex128}

def chm(dtm, dsm, chm, blocksize=256, thresh=None):
    
    """ 
    Create a CHM by dsm - dtm, done in blocks for efficiency
    
    Parameters 
    ----------- 
    
    dtm: string
              the dtm path
        
    dsm: string
            the dsm path
    
    chm: string
            the dsm path
        
    blocksize: int
                the chunk of raster read in & write out
    
    thresh: float
                set a minimum threshold (metres) at which canopy will be masked out
                e.g. 0.1

    """
    
    # Formerly - but not good if rasters dims don't match
    # #dtm
    # inDataset = gdal.Open(dtm)
    # dtmRas = inDataset.GetRasterBand(1)
    # #get no data as they may be different between rasters
    # dtmnd = dtmRas.GetNoDataValue()
    
    # #dsm
    # dsmin = gdal.Open(dsm)
    # dsmRas = dsmin.GetRasterBand(1)
    # dsmnd = dsmRas.GetNoDataValue()
    
    # quick and dirty way of ensuring alignment (not usually guaranteed)
    tmpstk = os.path.join(os.path.split(dsm)[0],'tmpstk.tif')
    _merge(names=[dsm, dtm], out_file=tmpstk)
    
    # ootdtype
    rds = gdal.Open(tmpstk)
    rdsDtype = rds.GetRasterBand(1).DataType
    
    # chm - must be from the dsm otherwise we get an offset
    outDataset = _copy_dataset_config(rds,  outMap=chm,
                         dtype=rdsDtype, bands=1)
    
    chmRas = outDataset.GetRasterBand(1)
    
    bnnd = rds.GetRasterBand(1)
    cols = rds.RasterXSize
    rows = rds.RasterYSize

    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize

        
    for i in tqdm(range(0, rows, blocksizeY)):
        if i + blocksizeY < rows:
            numRows = blocksizeY
        else:
            numRows = rows -i
    
        for j in range(0, cols, blocksizeX):
            if j + blocksizeX < cols:
                numCols = blocksizeX
            else:
                numCols = cols - j
            
            img = rds.ReadAsArray(j, i, numCols, numRows)
            
            # former code now redundant
            # dsarr = dsmRas.ReadAsArray(j, i, numCols, numRows)
            # nodata
            #dsarr[dsarr==dsmnd]=0
            #dtmarray = dtmRas.ReadAsArray(j, i, numCols, numRows)
            # nodata
            #dtmarray[dtmarray==dtmnd]=0
            
            #chm
            chm = img[0, :, :] - img[1, :, :]
            # in case of minus vals / silly no data
            chm[chm < 0] = 0 
            # to ensure there is no leak of dsm vals where underlying dtm is 0
            chm[img[1, :, :]==0]=0
            if thresh is not None:
                chm[chm<=thresh]=0
            chmRas.WriteArray(chm, j, i)
    
    os.remove(tmpstk)
    outDataset.FlushCache()
    outDataset = None

def _quickwarp(inRas, outRas, proj='EPSG:27700', **kwargs):
    
    """
    gdalwarp a dataset
    
    Parameters
    ----------
    
    inRas: string
            input raster
    
    outRas: string
            output raster
            
    proj: string
            an espg code e.g. 'EPSG:27700'
    
    Notes
    -----
    
    kwargs could include
    
    resampleAlg='bilinear' if interp or 'average' if aggregating
    
    xRes=10, yRes=10, multithread=True
    """
    ootRas = gdal.Warp(outRas, inRas, dstSRS=proj, format='Gtiff', 
                       callback=gdal.TermProgress, **kwargs)
    ootRas.FlushCache()
    ootRas=None

def mask_raster_multi(inputIm,  mval=1, rule='==', outval=None, mask=None,
                    blocksize = 256, FMT = None, dtype=None):
    """ 
    Perform a numpy masking operation on a raster where all values
    corresponding to, less than or greater than the mask value are retained 
    - does this in blocks for efficiency on larger rasters
    
    Parameters 
    ----------- 
    
    inputIm: string
              the granule folder 
        
    mval: int
           the masking value that delineates pixels to be kept
    
    rule: string
            the logic for masking either '==', '<' or '>'
        
    outval: numerical dtype eg int, float
              the areas removed will be written to this value default is 0
        
    mask: string
            the mask raster to be used (optional)
        
    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
    mode: string
           None > 10m data, '20' >20m
        
    blocksize: int
                the chunk of raster read in & write out

    """
    
    if FMT == None:
        FMT = 'Gtiff'
        fmt = '.tif'
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'
    
    
    if outval == None:
        outval = 0
    
    inDataset = gdal.Open(inputIm, gdal.GA_Update)
    bands = inDataset.RasterCount
    
    bnnd = inDataset.GetRasterBand(1)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize


    # So with most datasets blocksize is a row scanline
    if blocksize == None:
        blocksize = bnnd.GetBlockSize()
        blocksizeX = blocksize[0]
        blocksizeY = blocksize[1]
    else:
        blocksizeX = blocksize
        blocksizeY = blocksize
    
    if mask != None:
        msk = gdal.Open(mask)
        maskRas = msk.GetRasterBand(1)
        
        for i in tqdm(range(0, rows, blocksizeY)):
            if i + blocksizeY < rows:
                numRows = blocksizeY
            else:
                numRows = rows -i
        
            for j in range(0, cols, blocksizeX):
                if j + blocksizeX < cols:
                    numCols = blocksizeX
                else:
                    numCols = cols - j
                mask = maskRas.ReadAsArray(j, i, numCols, numRows)
                if mval not in mask:
                    array = np.zeros(shape=(numRows,numCols), dtype=np.int32)
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(band)
                        bnd.WriteArray(array, j, i)
                else:
                    
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(band)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        if rule == '==':
                            array[array != mval]=0
                        elif rule == '<':
                            array[array < mval]=0
                        elif rule == '>':
                            array[array > mval]=0
                        bnd.WriteArray(array, j, i)
                        
    else:
             
        for i in tqdm(range(0, rows, blocksizeY)):
                if i + blocksizeY < rows:
                    numRows = blocksizeY
                else:
                    numRows = rows -i
            
                for j in range(0, cols, blocksizeX):
                    if j + blocksizeX < cols:
                        numCols = blocksizeX
                    else:
                        numCols = cols - j
                    for band in range(1, bands+1):
                        bnd = inDataset.GetRasterBand(1)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        if rule == '==':
                            array[array != mval]=0
                        elif rule == '<':
                            array[array < mval]=0
                        elif rule == '>':
                            array[array > mval]=0
                        if outval != None:
                            array[array == mval] = outval     
                            bnd.WriteArray(array, j, i)

           
        inDataset.FlushCache()
        inDataset = None

def mask_with_poly(inshp, inras, layer=True, value=0, mtype='inside'):
    
    """ 
    Change raster values inside a polygon and update the raster
    
    Geometries must intersect!!!
    
    If done with layer=True entire raster will be read in at once and
    masked with layer, otherwise (False), each geometry will be read seperately
    and only that area read in. 
    
    Parameters
    ----------
    
    vector_path: string
                  input shapefile
        
    raster_path: string
                  input raster
    
    layer: bool
           whether to use the entire vector file as a mask (True) or loop through
           geometries seperately (False)
    
    value: int
            the value to alter
    
    mtype: str
            either area 'inside' or 'outside' of polygon is masked
    """    
    
    rds = gdal.Open(inras, gdal.GA_Update)
    rgt = rds.GetGeoTransform()
    bands = rds.RasterCount
    
    vds = ogr.Open(inshp, 1)  
    vlyr = vds.GetLayer(0)

    mem_drv = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('MEM')

    # Loop through vectors
    features = np.arange(vlyr.GetFeatureCount())
    
    rds_ext, spref, ext = _raster_extent2poly(inras)
    
    if layer == True:
        
        # What a mission - but more efficient for sure
        # TODO make clip shp with shp func from this
        ootds, ootlyr = _extent2lyr(inras, polytype='Memory')
        clipds, cliplyr = create_ogr_poly('out', spref.ExportToWkt(),
                                 file_type="Memory", field="id", 
                                 field_dtype=0)
        #self method result
        ogr.Layer.Clip(vlyr, ootlyr, cliplyr) # it works.....
        
        # debug
        #poly1 = loads(rds_ext.ExportToWkt())
        #feat = cliplyr.GetFeature(0)
        # geom2 = feat.GetGeometryRef()
        #wkt=geom2.ExportToWkt()
        #poly2 = loads(wkt)
        
        # dataset to put the rasterised into
        rvlyr = _copy_dataset_config(rds, FMT='MEM', outMap='copy',
                                   dtype=gdal.GDT_Byte, bands=1)

        gdal.RasterizeLayer(rvlyr, [1], cliplyr, burn_values=[1])
        rv_array = rvlyr.ReadAsArray()
        src_array = rds.ReadAsArray()
        # broadcast to 3d - nice!
        d_mask = np.broadcast_to(rv_array==1, src_array.shape)
        if mtype == 'inside':
            src_array[d_mask==1]=value
        elif mtype == 'outside':
            src_array[d_mask!=1]=value       
        rds.WriteArray(src_array)
    
    else:
        # TODO perhaps get rid or adapt to where a select poly is wanted
    
        for label in tqdm(features):
            feat = vlyr.GetNextFeature()
    
            if feat is None:
                continue
            geom = feat.geometry()
            
            # the poly may be partially outside the raster
            if rds_ext.Intersects(geom) == False:
                continue
            interpoly = rds_ext.Intersection(geom)
    
            src_offset = _bbox_to_pixel_offsets(rgt, interpoly)
            
            # calculate new geotransform of the feature subset
            new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5])
    
                
            # Create a temporary vector layer in memory
            mem_ds = mem_drv.CreateDataSource('out')
            mem_layer = mem_ds.CreateLayer('poly', None, ogr.wkbPolygon)
            featureDefn = mem_layer.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            feature.SetGeometry(interpoly)
            mem_layer.CreateFeature(feature)
            #mem_layer.CreateFeature(feat.Clone()) # if were using orig geom
    
            # Rasterize it
    
            rvds = driver.Create('', src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
         
            rvds.SetGeoTransform(new_gt)
            rvds.SetProjection(rds.GetProjectionRef())
            rvds.SetGeoTransform(new_gt)
            gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
            rv_array = rvds.ReadAsArray()
            
            # This could get expensive mem wise for big images
            for band in range(1, bands+1):
                bnd = rds.GetRasterBand(band)
                src_array = bnd.ReadAsArray(src_offset[0], src_offset[1], src_offset[2],
                                   src_offset[3])
                
                src_array[rv_array>0]=value
                bnd.WriteArray(src_array, src_offset[0], src_offset[1])
            
    rds.FlushCache()
        

    vds = None
    rds = None

def _extent2lyr(infile, filetype='raster', outfile=None, 
                polytype="ESRI Shapefile", geecoord=False, lyrtype='ogr'):
    
    """
    Get the coordinates of a files extent and return an ogr polygon ring with 
    the option to save the  file
    
    
    Parameters
    ----------
    
    infile: string
            input ogr compatible geometry file or gdal raster
            
    filetype: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    outfile: string
            the path of the output file, if not specified, it will be input file
            with 'extent' added on before the file type
    
    polytype: string
            ogr comapatible file type (see gdal/ogr docs) default 'ESRI Shapefile'
            ensure your outfile string has the equiv. e.g. '.shp' or in case of 
            memory only 'Memory' (outfile would be None in that case)
    
    geecoord: bool
           optionally convert to WGS84 lat,lon
    
    lyrtype: string
            either 'gee' which means earth engine or 'ogr' which returns ds and lyr
           
    Returns
    -------
    
    a GEE polygon geometry or ogr dataset and layer
    
    """
    # gdal/ogr read in etc
    if filetype == 'raster':
        _, rstref, ext = _raster_extent2poly(infile)
        # where ext is 
        xmin, ymin, xmax, ymax = ext # readable
        
    else:
        # tis a vector
        vds = ogr.Open(infile)
        lyr = vds.GetLayer()
        ext = lyr.GetExtent()
    
    # make the linear ring -
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmin, ymin) # the 5th is here as the ring must be closed
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    if geecoord == True:
        # Getting spatial reference of input 
        srs = lyr.GetSpatialRef()
    
        # make WGS84 projection reference3
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
    
        # OSR transform
        transform = osr.CoordinateTransformation(srs, wgs84)
        # apply
        poly.Transform(transform)
        
        tproj = wgs84
    if filetype == 'raster':
        tproj = rstref
    else:
        tproj = lyr.GetSpatialRef()
    
    # in case we wish to write it for later....    
#    if outfile != None:
#        outfile = infile[:-4]+'extent.shp'
    
    out_drv = ogr.GetDriverByName(polytype)
    
    # remove output shapefile if it already exists
    if outfile != None and polytype != 'Memory':
        if os.path.exists(outfile):
            out_drv.DeleteDataSource(outfile)
        ootds = out_drv.CreateDataSource(outfile)
    else:
        ootds = out_drv.CreateDataSource('out')

    ootlyr = ootds.CreateLayer("extent", tproj, geom_type=ogr.wkbPolygon)
    
    # add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    ootlyr.CreateField(idField)
    
    # create the feature and set values
    featureDefn = ootlyr.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(poly)
    feature.SetField("id", 1)
    ootlyr.CreateFeature(feature)
    feature = None
    
    # Save and close if not a memory driver
    
    ootds.FlushCache()
    
    if outfile != None and polytype != 'Memory':
        ootds = None
    
    if lyrtype == 'gee':
        poly.FlattenTo2D()
        return poly
    elif lyrtype == 'ogr':
        return ootds, ootlyr

def rasterize(inShp, inRas, outRas, field=None, fmt="Gtiff"):
    
    """ 
    Rasterize a polygon to the extent & geo transform of another raster


    Parameters
    -----------   
      
    inRas: string
            the input image 
        
    outRas: string
              the output polygon file path 
        
    field: string (optional)
             the name of the field containing burned values, if none will be 1s
    
    fmt: the gdal image format
    
    """
    
    
    
    inDataset = gdal.Open(inRas)
    
    # the usual 
    
    outDataset = _copy_dataset_config(inDataset, FMT=fmt, outMap=outRas,
                         dtype = gdal.GDT_Int32, bands=1)
    
    
    vds = ogr.Open(inShp)
    lyr = vds.GetLayer()
    
    
    if field == None:
        gdal.RasterizeLayer(outDataset, [1], lyr, burn_values=[1])
    else:
        gdal.RasterizeLayer(outDataset, [1], lyr, options=["ATTRIBUTE="+field])
    
    outDataset.FlushCache()
    
    outDataset = None
    
def polygonize(inRas, outPoly, outField=None,  mask=True, band=1, 
               filetype="ESRI Shapefile"):
    
    """ 
    Polygonise a raster

    Parameters
    -----------   
      
    inRas: string
            the input image   
        
    outPoly: string
              the output polygon file path 
        
    outField: string (optional)
             the name of the field containing burnded values

    mask: bool (optional)
            use the input raster as a mask

    band: int
           the input raster band
            
    """    
    
    #TODO investigate ways of speeding this up   

    options = []
    src_ds = gdal.Open(inRas)
    if src_ds is None:
        print('Unable to open %s' % inRas)
        sys.exit(1)
    
    try:
        srcband = src_ds.GetRasterBand(band)
    except RuntimeError as e:
        # for example, try GetRasterBand(10)
        print('Band ( %i ) not found')
        print(e)
        sys.exit(1)
    if mask == True:
        maskband = src_ds.GetRasterBand(band)
        options.append('-mask')
    else:
        mask = False
        maskband = None
    
    #    srs = osr.SpatialReference()
    #    srs.ImportFromWkt( src_ds.GetProjectionRef() )
    
    ref = src_ds.GetSpatialRef()
    #  create output data
    dst_layername = outPoly
    drv = ogr.GetDriverByName(filetype)
    dst_ds = drv.CreateDataSource(dst_layername)
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = ref )
    
    if outField is None:
        dst_fieldname = 'DN'
        fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(dst_fieldname)

    
    else: 
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(outField)

    gdal.Polygonize(srcband, maskband, dst_layer, dst_field,
                    callback=gdal.TermProgress)
    dst_ds.FlushCache()
    
    srcband = None
    src_ds = None
    dst_ds = None
    

def normalise_raster(inras, outras, dst_min=0, dst_max=100):
    
    """
    Scale a single band raster to 0 - 100
    
    Parameters
    ----------
    
    inras: string
            input raster
        
    outras: string
             output raster
    
    dst_min: int/float
             min value
             
    dst_max: int/float
              max value

    """
    #TODO add flexibility to the out range and datatype
       
    img = raster2array(inras, bands=[1])
    
    rescl = rescale_intensity(img, in_range='image',
                      out_range=(dst_min, dst_max)).astype(np.uint8)
    dtype = gdal.GDT_Byte 
    array2raster(rescl, 1, inras, outras, dtype, FMT=None)
    
    
    #TODO this would have been quicker one assumes
    # int object not iterable wtf
    # tried string now says RuntimeError: Too many command options '.'
#    src_min = str(bnd.GetMinimum())
#    src_max = str(bnd.GetMaximum())
#    params = [src_min, src_max, str(dst_min), str(dst_max)]
#    ds = gdal.Translate(outras, rds, scaleParams=params)
#    ds = None
#    rds = None

def _raster_extent2poly(inras):
    
    """
    Parameters
    ----------
    
    inras: string
        input gdal raster (already opened)
    
    """
    rds = gdal.Open(inras)
    rgt = rds.GetGeoTransform()
    minx = rgt[0]
    maxy = rgt[3]
    maxx = minx + rgt[1] * rds.RasterXSize
    miny = maxy + rgt[5] * rds.RasterYSize
    ext = (minx, miny, maxx, maxy)
    spref = rds.GetSpatialRef()
    
    # make the linear ring -
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(minx, miny)
    ring.AddPoint(maxx, miny)
    ring.AddPoint(maxx, maxy)
    ring.AddPoint(minx, maxy)
    ring.AddPoint(minx, miny) # the 5th is here as the ring must be closed
    
    # drop the geom into poly object
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    return poly, spref, ext


def clip_raster(inRas, inShp, outRas, cutline=False, fmt='GTiff'):

    """
    Clip a raster
    
    Parameters
    ----------
        
    inRas: string
            the input image 
            
    inShp: string
              the input polygon file path 
        
    outRas: string (optional)
             the clipped raster
             
    cutline: bool (optional)
             retain raster values only inside the polygon       
            
   
    """
    
    vds = ogr.Open(inShp)
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    lyr = vds.GetLayer()
          
    rds_ext, spref, ext = _raster_extent2poly(inRas)
    
    ootds, ootlyr = _extent2lyr(inRas, polytype='Memory')
    clipds, cliplyr = create_ogr_poly('out', spref.ExportToWkt(),
                             file_type="Memory", field="id", 
                             field_dtype=0)
    
    ogr.Layer.Clip(lyr, ootlyr, cliplyr)
    
    wrng = cliplyr.GetExtent()
    extent = [wrng[0], wrng[2], wrng[1], wrng[3]]

    #self method result
     # it works.....
    
    # debug
    # cliplyr.GetExtent()
    #poly1 = loads(rds_ext.ExportToWkt())
    #feat = cliplyr.GetFeature(0)
    #geom2 = feat.GetGeometryRef()
    #wkt=geom2.ExportToWkt()
    #poly2 = loads(wkt)
    
    # VERY IMPORTANT - TO AVOID COMING ACROSS THIS AGAIN
    # The problem iootrns[0]s that the bounds are not in the 'correct' order for gdalWarp
    # if taken from GetExtent() - they should in fact be in shapely order


    print('cropping')
    ootds = gdal.Warp(outRas, rds, format=fmt, outputBounds=extent)
    
    if cutline == True:
        print('masking to cutline')
        #cutline == True:
        # TODO - this only works with original extent of input raster,
        # need cropped one as input
        
        # print('cropping')
        # ootds = gdal.Warp(outRas,
        #           rds,
        #           format=fmt, 
        #           outputBounds=extent)
        
        # NOT finished need to read outds back in for procedure below
        # ootds.FlushCache()
        # ootds = None
      
        dtpe = rds.GetRasterBand(1).DataType
        
        rvlyr = _copy_dataset_config(ootds, FMT='MEM', outMap='copy',
                                   dtype=gdal.GDT_Byte, bands=1)

        gdal.RasterizeLayer(rvlyr, [1], cliplyr, burn_values=[1])
        
        mskbnd = rvlyr.GetRasterBand(1)

        cols = ootds.RasterXSize
        rows = ootds.RasterYSize

        blocksizeX = 256
        blocksizeY = 256
        
        bands = ootds.RasterCount
        
        
        for i in tqdm(range(0, rows, blocksizeY)):
                if i + blocksizeY < rows:
                    numRows = blocksizeY
                else:
                    numRows = rows -i
            
                for j in range(0, cols, blocksizeX):
                    if j + blocksizeX < cols:
                        numCols = blocksizeX
                    else:
                        numCols = cols - j

                    array = ootds.ReadAsArray(j, i, numCols, numRows)
                    mask = mskbnd.ReadAsArray(j, i, numCols, numRows)
                    if len(array.shape)==2:
                        array[mask!=1]=0
                    else:
                        d_mask = np.broadcast_to(mask==1, array.shape)
                        array[d_mask!=1]=0
                    ootds.WriteArray(array, j, i)
                        
    ootds.FlushCache()
    ootds = None
    rds = None

def fill_nodata(inRas, maxSearchDist=5, smoothingIterations=1, 
                bands=[1]):
    
    """
    fill no data using gdal
    
    Parameters
    ----------
    
    inRas: string
              the input image 
            
    maxSearchDist: int
              max search dist to fill
        
    smoothingIterations: int 
             the clipped raster
             
    bands: list of ints
            the bands to process      
    
    """
    
    rds = gdal.Open(inRas, gdal.GA_Update)
    
    for band in tqdm(bands):
        bnd = rds.GetRasterBand(band)
        gdal.FillNodata(targetBand=bnd, maskBand=None, 
                         maxSearchDist=maxSearchDist, 
                         smoothingIterations=smoothingIterations)
    
    rds.FlushCache()
    
    rds=None
        

def _copy_dataset_config(inDataset, FMT = 'Gtiff', outMap = 'copy',
                         dtype = gdal.GDT_Int32, bands = 1):
    """Copies a dataset without the associated rasters.

    """

    
    x_pixels = inDataset.RasterXSize  
    y_pixels = inDataset.RasterYSize  
    geotransform = inDataset.GetGeoTransform()
    PIXEL_SIZE = geotransform[1]  

    x_min = geotransform[0]
    y_max = geotransform[3]
    # x_min & y_max are like the "top left" corner.
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()   
    #dtype=gdal.GDT_Int32
    driver = gdal.GetDriverByName(FMT)
    
    # Set params for output raster
    outDataset = driver.Create(
        outMap, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    outDataset.SetGeoTransform((
        x_min,    # 0
        PIXEL_SIZE,  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -PIXEL_SIZE))
        
    outDataset.SetProjection(projection)
    
    return outDataset

# TODO this needs replaced with something better
def smrf_params():
    
    """
    Parameters
    ----------
    None
    
    Returns
    -------
    
    Dataframe of the parameters from the Pingle2013 paper
    
    """
    
    
    params = {'Sample': {0: '1-1',
              1: '1-2',
              2: '2-1',
              3: '2-2',
              4: '2-3',
              5: '2-4',
              6: '3-1',
              7: '4-1',
              8: '4-2',
              9: '5-1',
              10: '5-2',
              11: '5-3',
              12: '5-4',
              13: '6-1',
              14: '7-1'},
             'Features': {0: 'Mixed vegetation and buildings on hillside',
              1: 'Mixed vegetation and buildings',
              2: 'Road with bridge',
              3: 'Bridge and irregular ground surface',
              4: 'Large, irregularly shaped buildings',
              5: 'Steep slopes with vegetation',
              6: 'Complex building',
              7: 'Large gaps in data, irregularly shaped buildings',
              8: 'Trains in railway yard',
              9: 'Data gaps, vegetation on moderate slopes',
              10: 'Steep, terraced slopes',
              11: 'Steep, terraced slopes',
              12: 'Dense ground cover',
              13: 'Large gap in data',
              14: 'Underpass'},
             'slope': {0: 0.2,
              1: 0.18,
              2: 0.12,
              3: 0.16,
              4: 0.27,
              5: 0.16,
              6: 0.08,
              7: 0.22,
              8: 0.06,
              9: 0.05,
              10: 0.13,
              11: 0.45,
              12: 0.05,
              13: 0.28,
              14: 0.13},
             'window': {0: 16,
              1: 12,
              2: 20,
              3: 18,
              4: 13,
              5: 8,
              6: 15,
              7: 16,
              8: 49,
              9: 17,
              10: 13,
              11: 3,
              12: 11,
              13: 5,
              14: 15},
             'threshold': {0: 0.45,
              1: 0.3,
              2: 0.6,
              3: 0.35,
              4: 0.5,
              5: 0.2,
              6: 0.25,
              7: 1.1,
              8: 1.05,
              9: 0.35,
              10: 0.25,
              11: 0.1,
              12: 0.15,
              13: 0.5,
              14: 0.75},
             'scalar': {0: 1.2,
              1: 0.95,
              2: 0.0,
              3: 1.3,
              4: 0.9,
              5: 2.05,
              6: 1.5,
              7: 0.0,
              8: 0.0,
              9: 0.9,
              10: 2.2,
              11: 3.8,
              12: 2.3,
              13: 1.45,
              14: 0.0}}
    df = pd.DataFrame(params)
    # error pdal doesn't like int64 for serialise...
    df['window'] = df['window'].astype(float)
    return df

def array2raster(array, bands, inRaster=None, outRas=None, dtype=6,
                 FMT=None, rgt=None, espg=None):
    
    """
    Save a raster from a numpy array using the geoinfo from another.
    
    Parameters
    ----------      
    array: np array
            a numpy array - if multiband it should be shape (bands, y, x),
            otherwise the standard np (y,x)
    
    bands: int
            the no of bands. 
    
    inRaster: string
               the path of a raster.
    
    outRas: string
             the path of the output raster, if None will be written as 
             newraster.tif in the present directory
    
    dtype: int 
            though you need to know what the number represents!
            a GDAL datatype (see the GDAL website) e.g gdal.GDT_Int32
    
    FMT: string 
           (optional) a GDAL raster format (see the GDAL website) eg Gtiff, HFA, KEA.
    
    rgt: list
        in absence of template raster a geotransform list must be supplied with
        [x_min, PIXEL_SIZE,  0, y_max,  0, -PIXEL_SIZE]
        
    
    """

    if FMT == None:
        FMT = 'Gtiff'
        
    if FMT == 'HFA':
        fmt = '.img'
    if FMT == 'KEA':
        fmt = '.kea'
    if FMT == 'Gtiff':
        fmt = '.tif'    
    
    
    # RGT reminder
    # rgt =  (x_min, pixx,  0, y_max,  0, -pixy)
    # the 0s represent rotations which are not normally applied 
    
    #from gdalarray reminder
    # if interleave == 'band':
    #     interleave = True
    #     xdim = 2
    #     ydim = 1
    #     banddim = 0
    # if pixel (eg 1 band)
    #     xdim = 1
    #     ydim = 0
    
    if inRaster is not None:
    
        inras = gdal.Open(inRaster, gdal.GA_ReadOnly)    
        
        x_pixels = inras.RasterXSize  
        y_pixels = inras.RasterYSize  
        rgt = inras.GetGeoTransform()
        projection = inras.GetProjection()
    
    
    if inRaster is None and rgt is not None:
        geotransform = rgt
        
        spref = osr.SpatialReference() 
        spref.ImportFromEPSG(espg)
        projection = spref.ExportToProj4()
        
        if len(array.shape) > 2:
            x_pixels = int(array.shape[2])
            y_pixels = int(array.shape[1])
        else:
            x_pixels = int(array.shape[1])
            y_pixels = int(array.shape[0])

          
    driver = gdal.GetDriverByName(FMT)
    
    if outRas is None:
        outRas = "newraster.tif"

    dataset = driver.Create(
        outRas, 
        x_pixels,
        y_pixels,
        bands,
        dtype)

    dataset.SetGeoTransform(rgt)    

    dataset.SetProjection(projection)
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
    else:
        dataset.WriteArray(array)
        
    dataset.FlushCache() 
    dataset=None

        
def raster2array(inRas, bands=[1]):
    
    """
    Read a raster and return an array, either single or multiband

    
    Parameters
    ----------
    
    inRas: string
                  input  raster 
                  
    bands: list
                  a list of bands to return in the array
    
    """
    rds = gdal.Open(inRas)
   
   
    if len(bands) ==1:
        # then we needn't bother with all the crap below
        inArray = rds.GetRasterBand(bands[0]).ReadAsArray()
        
    else:
        #   The nump and gdal dtype (ints)
        #   {"uint8": 1,"int8": 1,"uint16": 2,"int16": 3,"uint32": 4,"int32": 5,
        #    "float32": 6, "float64": 7, "complex64": 10, "complex128": 11}
        
        # a numpy gdal conversion dict - this seems a bit long-winded
        dtypes = {"1": np.uint8, "2": np.uint16,
              "3": np.int16, "4": np.uint32,"5": np.int32,
              "6": np.float32,"7": np.float64,"10": np.complex64,
              "11": np.complex128}
        rdsDtype = rds.GetRasterBand(1).DataType
        inDt = dtypes[str(rdsDtype)]
        
        inArray = np.zeros((rds.RasterYSize, rds.RasterXSize, len(bands)), dtype=inDt) 
        for idx, band in enumerate(bands):  
            rA = rds.GetRasterBand(band).ReadAsArray()
            inArray[:, :, idx]=rA
   
   
    return inArray


def pdal_load(incld, reader='las', rng_limit=None):
    
    # rather slow...
    read = "readers."+reader
    js = [{"type": read,"filename":incld}]
    
    pipeline = pdal.Pipeline(json.dumps(js))
    
    if rng_limit != None:
        js.insert(1, {"type":"filters.range", 
                   "limits":rng_limit})
    
    pipeline.execute()
    
    # this is a big messy list
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    
    return metadata, arrays

def pdal_write(inarray, outcld, writer='las'):
    
    write = "writers."+writer
    
    js = [{"type": write, "filename": outcld}]
    
    pipeline = pdal.Pipeline(json.dumps(js), arrays=inarray)
    
    pipeline.execute()

# here for ref
#def pdal_edit(incld, outcld, reader='readers.las'):
#    
#    # rather slow...
#    js = [{"type": reader,"filename":incld}]
#    
#    pipeline = pdal.Pipeline(json.dumps(js))
#    
#    pipeline.execute()
#    
#    # this is a big messy list
#    arrays = pipeline.arrays
#    #metadata = pipeline.metadata
#    
#    # how to access a field and change the value
#    arrays[0]['Classification'][arrays[0]['Classification']==2]=0
#    
#    js2 = [{"type": "writers.las", "filename": outcld}]
#    
#    pipeline2 = pdal.Pipeline(json.dumps(js2), arrays=arrays)
#    
#    pipeline2.execute()
# But far better to write a sep numpy script and call as pdal recommends

def pdal_hag_chm(incld, inRas=None, outcld=None, reader='las',
                 hag_type="hag_dem", writer='las', thresh="40",
                       replace_z=True):
    
    """ 
    Normalise point heights with a DTM or nearest neoghbour
    to produce height above ground (hag)
    
    The norm'd values are written to 'HeightAboveGround' unless you specify
    the replace_z as True
    
    Parameters
    ----------
    
    incld: string
                  input pointcloud (las or ply)
        
    inRas: string
                  input raster
        
    outcld: string
              input pointcloud (las or ply)
    
    thresh: string
            a threshold in metres at which points are removed (ie spurious heights)
    
    replace_z: bool
            replace the Z values with the height above ground/chm values
    
    hag_type: string
            the hag_type either had_dem (needs a dtm) or hag_nn
            (uses nn points to acertain height)
       
    """    

    if hag_type == "hag_dem":
        hgdict = {"type":"filters."+hag_type, "raster": inRas }
    elif hag_type == "hag_nn":
        hgdict = {"type":"filters."+hag_type}
    
    
    js=[incld, hgdict,]
    
    th = {"type": "filters.expression",
            "expression": "HeightAboveGround < "+thresh}
    
    if replace_z is True:
        
        js.append({
            "type":"filters.ferry",
            "dimensions":"HeightAboveGround=>Z"
            })
        
        js.append(th)
    
        js.append({
            "type": "writers."+writer,
            "filename": outcld})

        
    else:
        js.append(th)
        js.append({"type":"writers."+writer,
        "filename": outcld,
        "extra_dims":"HeightAboveGround=float32"})
    
    pipeline = pdal.Pipeline(json.dumps(js))
    
    pipeline.execute()
    
    # Formerly this for replace z
    # # This seems a terrible way to do this - assume there is a direct
    # # way of writing to Z dim as above??
    # _, arrays = pdal_load(outcld, reader=reader)
    # arrays[0]['Z'] = arrays[0]['HeightAboveGround']
    # pdal_write(arrays, outcld, writer=writer)
        



def pdal_buildings(incld, outcld, inRas, knn=10, height="2", writer='las'):
    
    """
    Using the coplanar filter - unlikely to be that effective - will require
    
    lots of further cleaning
    
    Parameters
    ----------
    
    incld: string
                  input pointcloud (las or ply)
        
    inRas: string
                  input raster
        
    outcld: string
              input pointcloud (las or ply)
    
    dtm: string
            a dtm to derive height above ground from
    
    height: string
            the min height of a building
    
    writer: string
            the output format for pdal to write
    
    
    """
    
    js =  [incld,
            {
                "type": "filters.assign",
                "assignment": "Classification[:]=0"
    },
    {
        "type":"filters.hag_dem",
        "raster": inRas
    },
    {
      "type": "filters.range",
      "limits": "HeightAboveGround["+height+":)"
    },
    {
      "type": "filters.approximatecoplanar",
      "knn": knn
    },
    {
      "type": "filters.range",
      "limits": "Coplanar[1:1]"
    },
    
    {
        "type":"writers."+writer,
        "filename": outcld
    }
    ]
    
    if writer == 'ply':
        js[6]["precision"] = "6"
    
    pipeline = pdal.Pipeline(json.dumps(js))
    
    pipeline.execute()


def nrm_point_dtm(incld, inRas, outcld=None, reader='las', writer='las'):#,
                  #rng_limit="Classification[2:2]"):
    
    """ 
    Normalise point heights with a DTM / height relative to ground
    
    Now redundant - use pdal_hag_chm
    
    Parameters
    ----------
    
    incld: string
                  input pointcloud (las or ply)
        
    inRas: string
                  input raster
        
    outcld: string
              input pointcloud (las or ply)
    
    rng_limit: string
        only grid values in a range e.g. "Classification[1:2]" or "label[1:1]"
        as per the pdal convention
        
    """    
    
    
    # Now redundant - use the pdal hag version - good reminder of how to access
    # the arrays tho
    src = rasterio.open(inRas)
    
    _, arrays = pdal_load(incld, reader=reader)#, rng_limit=rng_limit)
    
    # not pretty but one line  
    # issue is for rasterio it needs to be an iterable
    coords = list(zip(arrays[0]['X'].tolist(), arrays[0]['Y'].tolist()))

    
    res = [x for x in tqdm(src.sample(coords))]
    # make continuous array
    gvals = np.concatenate(res, axis=0)
    
    arrays[0]['Z'] = arrays[0]['Z'] - gvals
    
    if outcld == None:
        outcld = incld
    
    pdal_write(arrays, outcld, writer=writer)
    


def drop_cld_value(incld, column='nrmz', rule='<', val=3, outcld=None,
                   remcld=False):
    
    """ 
    Delete point by threshhold eg less than 3m in height
    
    Parameters
    ----------
    
    incld: string
                  input pointcloud (las or ply)

    column: string
                  the attribute to threshold eg 'z'
        
    rule: string
            the logic for masking either '==', '<' or '>'

    val: string
            the value for removal
        
    outcld: string
              input pointcloud (las or ply)
    
    remcld: bool
            output the remaining points as a sperate cloud
        
    """    

    pcd = PyntCloud.from_file(incld)
    
    df = pcd.points
    
    if rule == '<':
        df = df.drop(df[df[column]<val].index)
    elif rule == '>':
        df = df.drop(df[df[column]>val].index)    
    elif rule == '==':
        df = df.drop(df[df[column]==val].index)
    pcd.points = df
    if outcld == None:
        outcld=incld
    pcd.to_file(outcld)
    
    if remcld ==True:
        rem = outcld[:-4]+'_rem.ply'
        pcd.to_file(rem)
    
def zonal_points(inShp, inRas, field, band=1, nodata_value=0, write_stat=True):
    
    """ 
    Get the pixel val at a given point and write to vector
    
    Parameters
    ----------
    
    inShp: string
                  input shapefile
        
    inRas: string
                  input raster
    
    field: string
                    the name of the field

    band: int
           an integer val eg - 2
                            
    nodata_value: numerical
                   If used the no data val of the raster
        
    """    
    
   

    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    rb = rds.GetRasterBand(band)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(nodata_value)
        rb.SetNoDataValue(nodata_value)

    vds = ogr.Open(inShp, 1)  # TODO maybe open update if we want to write stats
    vlyr = vds.GetLayer(0)
    
    if write_stat != None:
        # if the field exists leave it as ogr is a pain with dropping it
        # plus can break the file
        if _fieldexist(vlyr, field) == False:
            vlyr.CreateField(ogr.FieldDefn(field, ogr.OFTReal))
    
    
    
    feat = vlyr.GetNextFeature()
    features = np.arange(vlyr.GetFeatureCount())
    
    for label in tqdm(features):
    
            if feat is None:
                continue
            
            # the vector geom
            geom = feat.geometry()
            
            #coord in map units
            mx, my = geom.GetX(), geom.GetY()  

            # Convert from map to pixel coordinates.
            # No rotation but for this that should not matter
            px = int((mx - rgt[0]) / rgt[1])
            py = int((my - rgt[3]) / rgt[5])
            
            
            src_array = rb.ReadAsArray(px, py, 1, 1)

            if src_array is None:
                # unlikely but if none will have no data in the attribute table
                continue
            outval =  int(src_array.max())
            
#            if write_stat != None:
            feat.SetField(field, outval)
            vlyr.SetFeature(feat)
            feat = vlyr.GetNextFeature()
        
    if write_stat != None:
        vlyr.SyncToDisk()



    vds = None
    rds = None

def image_thresh(image):

#    image = rgb2gray(io.imread(im))
    
    if image.shape[0] > 4000:
        image = rescale(image, 0.5, preserve_range=True, anti_aliasing=True)
        image = np.uint8(image)
    
    def threshold(image, t):
        arr = da.from_array(image, chunks=image.shape)
        return arr > t
    
    all_thresholds = da.stack([threshold(image, t) for t in np.arange(255)])
    
    viewer = napari.view_image(image, name='input image')
    viewer.add_image(all_thresholds,
        name='thresholded', colormap='magenta', blending='additive'
    )

def apply_lut(src, lut):
    # dst(I) <-- lut(src)
    # https://stackoverflow.com/questions/14448763/
    #is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
    dst = lut[src]
    return dst

def colorscale(seg, prop='Area', custom=None):
    
    """
    Colour an array according to a region prop value
    
    Parameters
    ----------
    
    seg: np.array
        input array of labelled image
    
    prop: string
        sklearn region prop
    
    custom: lhedgeras)ist
            a custom list of values to apply to array
    
    Returns
    -------
    
    np array of attributed regions
    
    """
    
    if custom == None:
        props = regionprops(np.int32(seg))
         
        
        alist = [p[prop] for p in props] 
    
    else:
        alist = custom
    
    # there must be a zero to represent no data so insert at start
    alist.insert(0, 0)
    alist = np.array(alist)
    
    oot = apply_lut(seg, alist)
    
    return oot

def raster_skel(inras, outras, nodata=0, prune_len=70, blur=True, plot=None):
    
    """
    
    
    Parameters
    ----------
    
    inras: string
        input raster
    
    outras: string
        output raster
    
    nodata: int
            nodata value
    
    blur: bool
            use a blur to smooth binary
    
    prune_len: int
              length of line to be pruned from initial skeleton
    
    plot: string
          either 'plot' for debug/results or None
    
    Returns
    -------
    
    binary image of skeleton
    """
    
    img = raster2array(inras)
    
    img[img==nodata]=0
    
    # this step may not help actually
    if blur != False:
        img = cv2.GaussianBlur(img, (0,0), sigmaX=3, sigmaY=3, 
                               borderType=cv2.BORDER_DEFAULT)

    pcv.params.debug = plot
    
    skel = pcv.morphology.skeletonize(img)
    
    # input skel must be uint8!!! - slow but effective
    #pruned_skeleton, segmented_img, segment_objects
    
    #pcv.params.line_thickness = 3 
    pruned_skeleton, _, _ = pcv.morphology.prune(skel_img=skel, size=prune_len)
    
    # get the total length
    #np.count_nonzero(pruned_skeleton)*0.5/1000
    
    pruned_skeleton[pruned_skeleton>0]=1
    
    array2raster(pruned_skeleton, 1, inras, outras, 1)
    
    return pruned_skeleton
    

def las2ply_cc(incld):
    
    """
    Translate to ply 
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    
    outcld: string
            output cloud
    """

    
    cmd = ["cloudcompare.CloudCompare", "-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT",
           "PLY", "-O", "-GLOBAL_SHIFT", "AUTO", incld, "-SAVE_CLOUDS"]

    call(cmd)

    
def cldcompare_merge(folder, fmt='las'):
    
    """
    Merge ply/las clouds using cloud compare command line via subprocess
    
    Paramaters
    ----------
    
    folder: string
            the input directory
    
    fmt: string
        the format e.g 'las' or 'ply'
    """
    
    wcard ="*." + fmt 
    
    filelist = glob(os.path.join(folder, wcard))
    filelist.sort()
    
    # There should be a better way in bash, though it may be the cc cmd line 
    # also
    # Remember - No trailing spaces in any command as it will not work
    inp = "-o"
    #shft = "-GLOBAL_SHIFT AUTO" # not sure if this does anything
    
    if fmt == 'las':
      cf1 = "-C_EXPORT_FMT"
      cf2 = "LAS"
    else:
      cf1 = "-C_EXPORT_FMT"
      cf2 = "PLY"
    
    cmd = []
    
    for f in filelist:
        cmd.append(inp)
        cmd.append(f)
        #cmd.append(shft)
    # here sorting will not work - leave unordered
    #cloudcompare.CloudCompare -SILENT -NO_TIMESTAMP -C_EXPORT_FMT PLY ${files} -MERGE_CLOUDS
    cmd.insert(0, "cloudcompare.CloudCompare")
    cmd.insert(1,"-SILENT")
    cmd.insert(2,"-NO_TIMESTAMP")
    cmd.insert(3, cf1)
    cmd.insert(4, cf2)
    #cmd.insert(3,"-C_EXPORT_FMT PLY")
    cmd.append("-MERGE_CLOUDS")
    #cmd.append("-SAVE_CLOUDS FILE")
    #cmd.append(outcld)
    #cmd.append("-SAVE_MESHES ALL_AT_ONCE")
    call(cmd)
 


    
    
    
    
    
    
    
    
    
    

