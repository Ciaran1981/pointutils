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

gdal.UseExceptions()
ogr.UseExceptions()


# Notes on pdal pipelines
# To access the internals/outputs
#count = pipeline.execute()
#arrays = pipeline.arrays
#metadata = pipeline.metadata
#log = pipeline.log

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
    
    
    js = {"pipeline": [
    incld,
    {
        "type":"filters.crop",
        "polygon": poly.wkt
    },
    {
     "type":"writers."+writer,
     "filename":outcld
    }
    ]}
    
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
    
    outproj: string
            e.g. in form ESPG:4326 or proj4 if that doesn't work
            
    reader: string
            reader type (las, laz, ply, ept)
    
    writer: string
            writer type (las, laz, ply, ept)
    """
     
    
    js = [
        {
            "filename": incld,
            "type":"readers."+reader,
 #           "spatialreference": inproj,
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
    

def cloud_poly(incld, outshp=None, polytype="ESRI Shapefile"):
    
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
    
    Returns
    -------
    
    Shapely polygon
    
    """


    js = [incld, 
          {"type" : "filters.hexbin"}]
    
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
        spref = metajson['metadata']['readers.las']['spatialreference']
        
        proj = osr.SpatialReference()
        proj.ImportFromWkt(spref)
        
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
    
    

def intersect_clip(incld, inshp, outcld, column, index):
    
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
    
    """
    
    cldpoly = cloud_poly(incld)
    
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
        clip_cloud(incld, inshp, outcld, column, index)
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
        "type":"filters.overlay",
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

def pdal_denoise(incld, outcld, method="statistical", multi=3, k=8):
    
    """
    Denoise a pointcloud attribute using pdals outlier filters
    
    Parameters
    ----------
    
    incld: string
            input cloud
    
    outcld: string
            output cloud
    
    multi: int
            multiplier
    
    k: int
            k neighbours
            
    """
    
    js= {
        "pipeline": [
            incld,
            {
                "type": "filters.outlier",
                "method": method,
                "multiplier": multi,
                "mean_k": k
            },
            {
                "type": "writers.las",
                "filename": outcld
            }
        ]
        }
    
    pipeline = pdal.Pipeline(json.dumps(js))
    count = pipeline.execute()

def pdal_thin(incld, outcld, method="filters.sample", radius=5):
    
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
            
    """
    
    
    js = {"pipeline": [
            incld, {
        "type": method,
        "radius": radius
        },
        {
        "type":"writers.las",
        "filename":outcld
        }
        ]}

    
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
            scaling factor (def 1.25)
    
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

def chm(dtm, dsm, chm, blocksize = 256, FMT = None, dtype=None):
    
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
        
    FMT: string
          the output gdal format eg 'Gtiff', 'KEA', 'HFA'
        
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
    
    
    #dtm
    inDataset = gdal.Open(dtm, gdal.GA_Update)
    dtmRas = inDataset.GetRasterBand(1)
    #get no data as they may be different between rasters
    dtmnd = dtmRas.GetNoDataValue()
    
    #dsm
    dsmin = gdal.Open(dsm)
    dsmRas = dsmin.GetRasterBand(1)
    dsmnd = dsmRas.GetNoDataValue()
    
    # ootdtype
    rdsDtype = dsmRas.DataType
    
    # chm - must be from the dsm otherwise we get an offset
    outDataset = _copy_dataset_config(dsmin,  outMap=chm,
                         dtype=rdsDtype, bands=1)
    
    chmRas = outDataset.GetRasterBand(1)
    
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
            dsarr = dsmRas.ReadAsArray(j, i, numCols, numRows)
            # nodata
            dsarr[dsarr==dsmnd]=0
            dtmarray = dtmRas.ReadAsArray(j, i, numCols, numRows)
            # nodata
            dtmarray[dtmarray==dtmnd]=0
            #chm
            chm = dsarr - dtmarray
            chm[chm < 0] = 0 # in case of minus vals
            chmRas.WriteArray(chm, j, i)
            
    outDataset.FlushCache()
    outDataset = None

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

def clip_raster(inRas, inShp, outRas, cutline=True):

    """
    Clip a raster
    
    Parameters
    ----------
        
    inRas: string
            the input image 
            
    outPoly: string
              the input polygon file path 
        
    outRas: string (optional)
             the clipped raster
             
    cutline: bool (optional)
             retain raster values only inside the polygon       
            
   
    """
    

    vds = ogr.Open(inShp)
           
    rds = gdal.Open(inRas, gdal.GA_ReadOnly)
    
    lyr = vds.GetLayer()

    
    extent = lyr.GetExtent()
    
    extent = [extent[0], extent[2], extent[1], extent[3]]
            

    print('cropping')
    ootds = gdal.Warp(outRas,
              rds,
              format = 'GTiff', outputBounds = extent)
              
        
    ootds.FlushCache()
    ootds = None
    rds = None
    
    if cutline == True:
        
        rds1 = gdal.Open(outRas, gdal.GA_Update)
        rasterize(inShp, outRas, outRas[:-4]+'mask.tif', field=None,
                  fmt="Gtiff")
        
        mskds = gdal.Open(outRas[:-4]+'mask.tif')
        
        mskbnd = mskds.GetRasterBand(1)

        cols = mskds.RasterXSize
        rows = mskds.RasterYSize

        blocksizeX = 256
        blocksizeY = 256
        
        bands = rds1.RasterCount
        
        mskbnd = mskds.GetRasterBand(1)
        
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
                        
                        bnd = rds1.GetRasterBand(band)
                        array = bnd.ReadAsArray(j, i, numCols, numRows)
                        mask = mskbnd.ReadAsArray(j, i, numCols, numRows)
                        
                        array[mask!=1]=0
                        bnd.WriteArray(array, j, i)
                        
        rds1.FlushCache()
        rds1 = None

def fill_nodata(inRas, outRas, maxSearchDist=5, smoothingIterations=1):
    
    """
    fill no data using gdal
    
    Parameters
    ----------
    
    inRas: string
              the input image 
            
    maxSearchDist: int
              the input polygon file path 
        
    smoothingIterations: int (optional)
             the clipped raster
             
    maskBand: bool (optional)
             the mask band for where to fill      
    
    """
    
    rds = gdal.Open(inRas, gdal.GA_Update)
    
    bnd = rds.GetRasterBand(1)
    
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

#def cldcompare_merge(folder):
#    
#    """
#    Merge ply clouds using cloud compare command line via subprocess
#    
#    Paramaters
#    ----------
#    
#    folder: string
#            the input directory
#            
#    """
#    
#    filelist = glob(os.path.join(folder, "*.ply"))
#    filelist.sort()
#    
#    # There should be a better way in bash, though it may be the cc cmd line 
#    # also
#    inp = "-o "
#    shft = " -GLOBAL_SHIFT AUTO"
#    
#    cmd = []
#    
#    for f in filelist:
#        cmd.append(inp)
#        cmd.append(f)
#        cmd.append(shft)
#    # here sorting will not work - leave unordered
#    #cloudcompare.CloudCompare -SILENT -NO_TIMESTAMP -C_EXPORT_FMT PLY ${files} -MERGE_CLOUDS
#    cmd.insert(0, "cloudcompare.CloudCompare")
#    cmd.insert(1,"-SILENT")
#    cmd.insert(2,"-NO_TIMESTAMP")
#    #cmd.insert(3,"-C_EXPORT_FMT PLY")
#    cmd.append("-MERGE_CLOUDS")
#    #cmd.append("-SAVE_MESHES ALL_AT_ONCE")
#    call(cmd)
#    #log = open(os.path.join(folder, 'log.txt'), "w")
#    #ret = call(cmd, log)
#    
#    if ret !=0:
#        print('A micmac error has occured - check the log file')
#        sys.exit()
    
    
    
    
    
    
    
    
    
    
    

