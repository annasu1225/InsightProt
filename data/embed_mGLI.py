#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:03:40 2022

@author: shenli

modified by Anna Su
Date: Dec 2025
"""
import numpy as np
from scipy.spatial import distance_matrix
from Bio import PDB
import argparse
import torch

def Gauss_linking_integral(a,b):
    '''
    a:tuple; elements are head and tail
    of line segment, each is a (3,) array representing the xyz coordinate.
    '''
    
    R = np.empty((2,2),dtype=tuple)
    for i in range(2):
        for j in range(2):
            R[i,j]=a[i]-b[j]
          
    n=[]
    cprod = []

    cprod.append(np.cross(R[0,0],R[0,1]))
    cprod.append(np.cross(R[0,1],R[1,1]))
    cprod.append(np.cross(R[1,1],R[1,0]))
    cprod.append(np.cross(R[1,0],R[0,0]))
    
    for c in cprod:
        n.append(c/(np.linalg.norm(c)+1e-6))
    
    area1 = np.arcsin(np.dot(n[0],n[1]))
    area2 = np.arcsin(np.dot(n[1],n[2]))
    area3 = np.arcsin(np.dot(n[2],n[3]))
    area4 = np.arcsin(np.dot(n[3],n[0]))
    
    sign = np.sign(np.cross(a[1]-a[0],b[1]-b[0]).dot(a[0]-b[0]))
    Area = area1+area2+area3+area4

    return sign*Area


def line_classification1(cloudpoints,center_index,r):
    distances = distance_matrix(cloudpoints,cloudpoints[center_index].reshape(1,-1))
    in_line =[]
    out_line=[]
    if distances[0]<r:
        end2 = (cloudpoints[0]+cloudpoints[1])/2
        end1 = cloudpoints[0]
        if center_index == 0:
            in_line.append((end1,end2))
        else:
            out_line.append((end1,end2))
        
    
    for i in range(1,cloudpoints.shape[0]-1):
        if distances[i]<r:
            if i!=center_index:
                out_line.append(((cloudpoints[i-1]+cloudpoints[i])/2,cloudpoints[i]))
                out_line.append((cloudpoints[i],(cloudpoints[i]+cloudpoints[i+1])/2))
            else:
                in_line.append(((cloudpoints[i-1]+cloudpoints[i])/2,cloudpoints[i]))
                in_line.append((cloudpoints[i],(cloudpoints[i]+cloudpoints[i+1])/2))
                
    n = cloudpoints.shape[0]-1   

    if distances[n]<r:   
        end1 = (cloudpoints[n-1]+cloudpoints[n])/2
        end2 = cloudpoints[n]
        if center_index == n:

            in_line.append((end1,end2))
        else:
            out_line.append((end1,end2))
    
    
    
    return out_line, in_line
    

def get_protein_ca_atom_coordinate(pdbid,filepath):
    parser = PDB.PDBParser()
    struct = parser.get_structure(
        pdbid,
        filepath
        )

    CA_coordinates = np.array([])
    
    for model in struct:
        labels=np.array([])
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.id == 'CA':
                       XYZ = atom.get_coord()
                       bfactor = atom.bfactor
                       labels= np.append(labels,bfactor)      
                       CA_coordinates = np.hstack([CA_coordinates,XYZ])                                    
        break
    return CA_coordinates.reshape(-1,3),labels


def GLI_feature_standard(CA_coor,r):
    GLI = []
    for i in range(CA_coor.shape[0]):
        outline,inline = line_classification1(CA_coor,i,r)
        L=0

        for line1 in outline:
            for line2 in inline:
                l = Gauss_linking_integral(line1,line2)
                L += np.abs(l)

        GLI.append(L)
        
    return GLI

def get_inteval(n=10,minimal=2,maximal=11):
    scales = np.linspace(minimal,maximal,n)
    intevals=[]
    for i in range(n-1):
        r1 = scales[i]
        r2 = scales[i+1]
        intevals.append([r1,r2])
    return intevals

def summarize_profile(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return np.zeros(8, dtype=float)
    q = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])
    return np.array([
        v.mean(),
        v.std(),
        v.min(),
        q[0], q[1], q[2], q[3], q[4],
        v.max(),
    ], dtype=float)

def generate_mGLI_feature(pdbid, pdb_path, n_scales=10, minimal=2, maximal=11):
    CA_coor, _ = get_protein_ca_atom_coordinate(pdbid, pdb_path)
    intervals = get_inteval(n=n_scales, minimal=minimal, maximal=maximal)

    emb_parts = []
    for r1, r2 in intervals:
        prof2 = np.array(GLI_feature_standard(CA_coor, r2), dtype=float)
        prof1 = np.array(GLI_feature_standard(CA_coor, r1), dtype=float)
        shell = prof2 - prof1
        emb_parts.append(summarize_profile(shell))
    return np.concatenate(emb_parts, axis=0)

def main():
    parser = argparse.ArgumentParser(description='Compute mGLI features for a PDB file.')
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file.')
    args = parser.parse_args()

    pdb_path = args.pdb_file
    pdb_id = pdb_path.split("/")[-1].split(".pdb")[0]
    features = generate_mGLI_feature(pdb_id, pdb_path)
    
    print("mGLI features:", features)
    print("Feature length:", len(features))

    output_path = pdb_path[:-4] + "_mGLI.pt"
    torch.save(features, output_path)

if __name__ == "__main__":
    main()