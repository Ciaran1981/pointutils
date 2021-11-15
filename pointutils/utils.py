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
from osgeo import gdal, ogr

    
def clip_cloud(incld, inshp, outcld, column, index, polyrng="[32670:32670]"):

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
    
    """

    gdf = gpd.read_file(inshp)
    
    row = gdf[gdf[column]==index]
    
    # if there is more than one, merge them
    if len(row) > 0:
         row = row.dissolve(by=column)
    
    wkt = row.geometry.to_wkt()
    
    inwkt = wkt.iloc[0]
    # poly1 = loads(inwkt)
    
    js = {"pipeline": [
    incld,
    {
        "type":"filters.crop",
        "polygon": inwkt
    },
    {
        "type":"writers.las",
        "filename":outcld
    }
    ]}

    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()        
    
def grid_cloud(incld, outfile, attribute="label", reader="readers.ply",
               writer="writers.gdal", spref="EPSG:21818", dtype="uint16_t",
               outtype='mean', resolution=0.1):
    
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
            dtype in pdal format (see pdal)
        
    outtype: string
            mean, min or max
            
    resolution: float
            in the unit required

    """
    
    #json from args
    js = {
          "pipeline":[
            {
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
            }
          ]
        }  
    
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()
    #log = pipeline.log
    
def grid_cloud_batch(folder, attribute="label", reader="readers.ply", 
                    writer="writers.gdal", spref="EPSG:21818", 
                    dtype="uint16_t", outtype='mean', resolution=0.1, nt=-1):
    
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
    
    
    plylist = glob(os.path.join(folder, '*.ply'))
    plylist.sort()

    outlist = [ply[:-3]+'tif' for ply in plylist]
        
        
    Parallel(n_jobs=nt, verbose=2)(delayed(grid_cloud)(ply, out, attribute,
             reader, writer, spref, dtype,
             outtype, resolution) for ply, out in zip(plylist, outlist))



def pdal_smrf(incld, smrf=None, scalar=1.25, slope=0.15, threshold=0.5, 
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
            scaling factor (def 1.25)
    
    slope: float
            slope thresh (def 0.15)
    
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
        params["type"] ="filters.smrf"

    else:
         params = {
            "type":"filters.smrf",
            "scalar":scalar,
            "slope":slope,
            "threshold":threshold,
            "window":window
        }
        
    js = [
        incld,
        params,
        {
            "type":"filters.range",
            "limits":"Classification"+clsrange
        },
        outcld
    ]
    
    pipeline = pdal.Pipeline(json.dumps(js))
    pipeline.execute()

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

def split_into_classes(incld, field='label', classes=None):
    
    """
    Split a classified ply file into seperate classes or output only those 
    selected
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    classes: list of ints
            a list of the classes to create seperate pointclouds with
    
    """
    
    pcd = PyntCloud.from_file(incld)
    
    #recall that pyntcloud child objects are linked, hence the loop
    
    if classes == None:
        classes = pcd.points[field].unique().tolist()
        classes.sort()
    
    for c in tqdm(classes):
        # must instantiate new one
        newpoints = PyntCloud(pcd.points[pcd.points[field]==c])
        name, ext = os.path.splitext(incld)
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
        # seem to lose rgb values
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
    
    
# TODO this needs replaced
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