# viewer_indexed_chunked.py
import sys
import os
import math
import struct
import ctypes
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# -------------------------
# VTK Import
# -------------------------
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("[Warning] VTK module not available. VTK file support will be disabled.")

# -------------------------
# Configurable limits
# -------------------------
# How many triangles per GPU chunk. Keep this high for fewer draw calls, low for low memory spikes.
TRIANGLES_PER_CHUNK = 300_000   # adjust up/down for your GPU (300k triangles -> 900k vertices)
DEFAULT_WINDOW = (1024, 768)
FPS_FONT_SIZE = 20 # Font size for the stats counter

# Slicing / steganography defaults
DEFAULT_NUM_SLICES = 64
DEFAULT_SLICE_AXIS = 'Z'
DEFAULT_BITS_PER_COORD = 1
DEFAULT_STEGO_SCALE = 1000000

# -------------------------
# Utilities
# -------------------------
def normalize_vertices_array(verts: np.ndarray):
    """
    Center and scale to unit bounding box. This is only a visual transform (keeps geometry intact).
    verts: (N,3) float32 array of vertex positions
    """
    if verts.size == 0:
        return verts
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2.0
    verts = verts - center
    scale = (maxs - mins).max()
    if scale == 0:
        return verts.astype(np.float32)
    verts = verts / scale
    return verts.astype(np.float32)

# -------------------------
# Parse STL into triangles (robust binary + ASCII)
# returns numpy array shape (T*3, 3)
# -------------------------
def load_stl_triangles(filename: str) -> np.ndarray:
    try:
        size = os.path.getsize(filename)
    except:
        size = 0

    # Try binary first (common)
    try:
        with open(filename, 'rb') as f:
            header = f.read(80)
            count_bytes = f.read(4)
            if len(count_bytes) < 4:
                raise ValueError("Too short for binary STL")

            tri_count = struct.unpack('<I', count_bytes)[0]
            expected = 80 + 4 + tri_count * 50
            # If the sizes mismatch widely, fallback
            if size != 0 and abs(expected - size) > 1000:
                raise ValueError("Binary header mismatch; fallback to ASCII")

            data = f.read(tri_count * 50)
            verts = []
            ptr = 0
            for i in range(tri_count):
                block = data[ptr:ptr+50]
                if len(block) < 50:
                    break
                # v1 12-24, v2 24-36, v3 36-48
                v1 = struct.unpack('<3f', block[12:24])
                v2 = struct.unpack('<3f', block[24:36])
                v3 = struct.unpack('<3f', block[36:48])
                verts.extend([v1, v2, v3])
                ptr += 50
            return np.array(verts, dtype=np.float32)
    except Exception:
        # ASCII fallback
        verts = []
        with open(filename, 'r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('vertex'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                        except:
                            pass
        return np.array(verts, dtype=np.float32)

# -------------------------
# Parse OBJ into triangles (fan triangulation for polygons)
# returns numpy array shape (T*3, 3)
# -------------------------
def load_obj_triangles(filename: str) -> np.ndarray:
    verts = []
    faces = []
    with open(filename, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    except:
                        pass
            elif line.startswith('f '):
                parts = line.split()[1:]
                idxs = []
                for p in parts:
                    try:
                        idx = int(p.split('/')[0])
                    except:
                        continue
                    if idx < 0:
                        idx = len(verts) + idx
                    else:
                        idx = idx - 1
                    idxs.append(idx)
                if len(idxs) >= 3:
                    for i in range(1, len(idxs)-1):
                        faces.append((idxs[0], idxs[i], idxs[i+1]))
    if len(faces) == 0:
        return np.zeros((0,3), dtype=np.float32)
    v_arr = np.array(verts, dtype=np.float32)
    tri_list = []
    for a,b,c in faces:
        if 0 <= a < len(verts) and 0 <= b < len(verts) and 0 <= c < len(verts):
            tri_list.append(v_arr[a]); tri_list.append(v_arr[b]); tri_list.append(v_arr[c])
    return np.array(tri_list, dtype=np.float32)

# -------------------------
# Parse VTK into triangles using VTK library
# returns numpy array shape (T*3, 3)
# -------------------------
def load_vtk_triangles(filename: str) -> np.ndarray:
    if not VTK_AVAILABLE:
        raise ImportError("VTK module not available. Please install vtk package.")
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()
    
    if not poly:
        # Try generic reader as fallback
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        
        if isinstance(data, vtk.vtkPolyData):
            poly = data
        else:
            # Convert to poly data if needed
            if isinstance(data, vtk.vtkStructuredPoints) or isinstance(data, vtk.vtkImageData):
                contour = vtk.vtkMarchingCubes()
                contour.SetInputData(data)
                contour.SetValue(0, 1)
                contour.Update()
                poly = contour.GetOutput()
            elif isinstance(data, vtk.vtkUnstructuredGrid):
                surface = vtk.vtkDataSetSurfaceFilter()
                surface.SetInputData(data)
                surface.Update()
                poly = surface.GetOutput()
            else:
                raise RuntimeError("Unsupported VTK dataset")
    
    # IMPORTANT: DO NOT re-triangulate or modify the mesh structure
    # Extract vertices in EXACT order as stored
    points = poly.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    # Get all vertices in original order
    vertices = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        vertices[i] = points.GetPoint(i)
    
    # Extract triangles exactly as stored
    tris = []
    cells = poly.GetPolys()
    cells.InitTraversal()
    idList = vtk.vtkIdList()
    
    while cells.GetNextCell(idList):
        n = idList.GetNumberOfIds()
        if n == 3:
            # Triangle
            tris.append([idList.GetId(0), idList.GetId(1), idList.GetId(2)])
        elif n > 3:
            # Polygon - use fan triangulation but preserve order
            for j in range(1, n-1):
                tris.append([idList.GetId(0), idList.GetId(j), idList.GetId(j+1)])
    
    # Create triangle vertex list
    tri_verts = []
    for tri in tris:
        tri_verts.append(vertices[tri[0]])
        tri_verts.append(vertices[tri[1]])
        tri_verts.append(vertices[tri[2]])
    
    return np.array(tri_verts, dtype=np.float32)
    
    # Triangulate
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    poly = tri.GetOutput()
    
    vertices = []
    indices = []
    
    points = poly.GetPoints()
    polys = poly.GetPolys()
    
    for i in range(points.GetNumberOfPoints()):
        vertices.extend(points.GetPoint(i))
    
    polys.InitTraversal()
    idList = vtk.vtkIdList()
    
    while polys.GetNextCell(idList):
        if idList.GetNumberOfIds() == 3:
            indices.extend([
                idList.GetId(0),
                idList.GetId(1),
                idList.GetId(2)
            ])
    
    # Convert indices to vertex list
    verts_array = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    indices_array = np.array(indices, dtype=np.int32)
    
    # Create triangle vertex list
    tri_verts = []
    for i in range(0, len(indices_array), 3):
        if i+2 < len(indices_array):
            idx1, idx2, idx3 = indices_array[i], indices_array[i+1], indices_array[i+2]
            if idx1 < len(verts_array) and idx2 < len(verts_array) and idx3 < len(verts_array):
                tri_verts.append(verts_array[idx1])
                tri_verts.append(verts_array[idx2])
                tri_verts.append(verts_array[idx3])
    
    return np.array(tri_verts, dtype=np.float32)

def make_indexed_mesh(vertices: np.ndarray, decimal_tol=6):
    """
    Deduplicate vertices by rounding to decimal_tol decimals.
    This preserves the exact geometry up to rounding tolerance used only for hashing; triangle indices remain identical as long as duplicates are exact.
    """
    if vertices.size == 0:
        return np.zeros((0,3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

    # Round for hashing (helps floating noise). You can increase decimals if you want stricter uniqueness.
    rounded = np.round(vertices, decimals=decimal_tol)
    # Use structured array trick for unique
    flat = rounded.view([('', rounded.dtype)] * rounded.shape[1]).squeeze()
    unique_vals, inverse = np.unique(flat, return_inverse=True)
    # unique_vals is structured; convert back
    unique_vertices = unique_vals.view(rounded.dtype).reshape(-1, 3).astype(np.float32)
    indices = inverse.astype(np.uint32)  # length = vertices.shape[0]
    return unique_vertices, indices

# -------------------------
# Compute per-vertex normals by averaging face normals (smooth shading).
# Input:
#   vertices (V,3) unique vertex positions
#   indices (N,) where N = number of triangle vertices (3 per triangle)
# Output:
#   normals (V,3) float32
# -------------------------
def compute_vertex_normals(vertices: np.ndarray, indices: np.ndarray):
    vcount = vertices.shape[0]
    normals = np.zeros((vcount, 3), dtype=np.float64)
    tris = indices.reshape(-1, 3)
    for a,b,c in tris:
        v1 = vertices[a]
        v2 = vertices[b]
        v3 = vertices[c]
        face = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(face)
        if norm > 0:
            face = face / norm
        normals[a] += face
        normals[b] += face
        normals[c] += face
    # normalize
    nlen = np.linalg.norm(normals, axis=1)
    nz = nlen > 0
    normals[nz] = normals[nz] / nlen[nz][:,None]
    # fallback zeros to up vector
    normals[~nz] = np.array([0.0, 0.0, 1.0])
    return normals.astype(np.float32)

# -------------------------
# GPU chunk holder: each chunk has VBO + NBO + EBO and an index count
# -------------------------
class MeshChunk:
    def __init__(self):
        self.vbo = None
        self.nbo = None
        self.ebo = None
        self.index_count = 0

    def upload(self, positions: np.ndarray, normals: np.ndarray, indices: np.ndarray):
        """
        positions: (P,3) float32
        normals:   (P,3) float32
        indices:   (M,) uint32  (indices refer to positions)
        """
        # cleanup
        self.release()
        # create VBO for positions
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # normals
        self.nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.nbo)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

        # EBO (element buffer)
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

        self.index_count = int(indices.size)

    def draw(self):
        if not self.vbo or not self.ebo or self.index_count == 0:
            return

        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glEnableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.nbo)
        glNormalPointer(GL_FLOAT, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # draw all indices - assumes GL_UNSIGNED_INT supported (most modern GPUs)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

        # unbind
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def release(self):
        try:
            if self.vbo:
                glDeleteBuffers(1, [self.vbo])
        except:
            pass
        try:
            if self.nbo:
                glDeleteBuffers(1, [self.nbo])
        except:
            pass
        try:
            if self.ebo:
                glDeleteBuffers(1, [self.ebo])
        except:
            pass
        self.vbo = None; self.nbo = None; self.ebo = None; self.index_count = 0

# -------------------------
# Model manager: builds indexed mesh, chunks it, computes normals, uploads
# -------------------------
class Model:
    def __init__(self):
        self.chunks = []   # list of MeshChunk
        self.vertex_count = 0
        self.triangle_count = 0
        self.positions = None
        self.normals = None
        self.indices = None
        self.streaming_mode = False
        self.disable_normalization = False

        # slice cache (list of 2D numpy arrays of points per slice)
        self.slice_points = None
        self.slice_points_3d = None  # list of (N_i, 3) arrays on the actual slice planes
        self.slice_axis = DEFAULT_SLICE_AXIS
        self.slice_count = DEFAULT_NUM_SLICES
        self.slice_planes = None      # 1D array of plane positions along slice_axis
        self.slice_bounds = None      # (mins, maxs) in model space for current slices

        # mask of vertices whose positions were modified by steganography
        self.changed_vertex_mask = None

    def clear_gpu(self):
        for c in self.chunks:
            try:
                c.release()
            except:
                pass
        self.chunks = []

    def load_from_file(self, filename: str):
        self.clear_gpu()
        ext = os.path.splitext(filename)[1].lower()
        self.original_ext = ext

        if ext == '.stl':
            tri_verts = load_stl_triangles(filename)
            
        elif ext == '.obj':
            tri_verts = load_obj_triangles(filename)
            
        elif ext == '.vtk':
            if not VTK_AVAILABLE:
                raise ImportError("VTK module not available. Please install vtk package.")
            tri_verts = load_vtk_triangles(filename)
            
        else:
            raise ValueError("Unsupported file type: " + ext)

        if tri_verts.size == 0:
            raise ValueError("No geometry found in file.")

        # tri_verts is a list of per-vertex positions: length = triangles*3
        total_triangles = tri_verts.shape[0] // 3
        print(f"[Model] Triangles: {total_triangles:,}, vertices (raw): {tri_verts.shape[0]:,}")

        # For VTK files, use more precise deduplication to preserve stego data
        if ext == '.vtk':
            # Use higher precision for VTK to preserve embedded bits
            verts_unique, indices = make_indexed_mesh(tri_verts, decimal_tol=10)
        else:
            verts_unique, indices = make_indexed_mesh(tri_verts, decimal_tol=6)

        # compute normals per unique vertex
        normals = compute_vertex_normals(verts_unique, indices)

        # Normalize scale for viewing (this does not alter counts or indices)
        # BUT: For VTK with embedded data, we need to be careful
        if ext == '.vtk':
            # Store original positions before normalization
            self.original_positions = verts_unique.copy()
            # Still normalize for display
            verts_unique = normalize_vertices_array(verts_unique)
        else:
            verts_unique = normalize_vertices_array(verts_unique)

        self.positions = verts_unique
        self.normals = normals
        self.indices = indices
        self.vertex_count = verts_unique.shape[0]
        self.triangle_count = indices.size // 3

        # Store file extension
        self.file_extension = ext
        
        # reset slice cache on new model
        self.slice_points = None
        self.slice_points_3d = None
        self.slice_axis = DEFAULT_SLICE_AXIS
        self.slice_count = DEFAULT_NUM_SLICES
        self.slice_planes = None
        self.slice_bounds = None

        # reset changed-vertex tracking on new model
        self.changed_vertex_mask = None
        self.stego_original_positions = None
        self.stego_traversal_order = None

        # build GPU chunks for the freshly loaded model
        self._build_chunks()

    def _build_chunks(self):
        
        """
        (Re)build GPU chunks from current positions/normals/indices.
        """
        self.clear_gpu()
        if self.positions is None or self.indices is None or self.indices.size == 0:
            return

        tris_total = self.triangle_count
        tris_per_chunk = TRIANGLES_PER_CHUNK
        if tris_per_chunk <= 0:
            tris_per_chunk = tris_total

        self.chunks = []
        tri_indices = self.indices.reshape(-1, 3)
        start_tri = 0
        try:
            while start_tri < tris_total:
                end_tri = min(tris_total, start_tri + tris_per_chunk)
                chunk_tris = tri_indices[start_tri:end_tri]  # shape (K,3)
                unique_idx, inv = np.unique(chunk_tris.flatten(), return_inverse=True)
                pos_chunk = self.positions[unique_idx]
                norm_chunk = self.normals[unique_idx]
                local_indices = inv.astype(np.uint32).reshape(-1)
                chunk = MeshChunk()
                chunk.upload(pos_chunk.astype(np.float32), norm_chunk.astype(np.float32), local_indices)
                self.chunks.append(chunk)
                start_tri = end_tri
        except Exception as e:
            print("[Model] GPU upload failed in _build_chunks - falling back to streaming mode:", e)
            for c in self.chunks:
                try:
                    c.release()
                except:
                    pass
            self.chunks = []
            self.streaming_mode = True

    # -------------------------------------------------
    # Steganography on in-memory vertex positions
    # -------------------------------------------------
    def _build_slice_vertex_indices_from_indices(self, axis=None, num_slices=None):
        """
        Build traversal order based ONLY on the vertex indices and triangle connectivity.
        This is completely independent of coordinate values, so it survives export/import.
        
        Strategy: Sort vertices by their first occurrence in the index buffer,
        then by their actual index number. This is deterministic and geometry-independent.
        """
        if self.indices is None or self.indices.size == 0:
            return []
        
        num_slices = num_slices or self.slice_count
        vertex_count = self.positions.shape[0] if self.positions is not None else 0
        
        if vertex_count == 0:
            return []
        
        # Find first occurrence of each vertex in the index buffer
        first_occurrence = np.full(vertex_count, -1, dtype=np.int32)
        for i, idx in enumerate(self.indices):
            if first_occurrence[idx] == -1:
                first_occurrence[idx] = i
        
        # Create vertex list sorted by first occurrence, then by index
        vertices = np.arange(vertex_count, dtype=np.int32)
        
        # Sort by: (first_occurrence, vertex_index)
        # Vertices that never appear get first_occurrence=-1, put them at the end
        mask_used = first_occurrence >= 0
        used_verts = vertices[mask_used]
        unused_verts = vertices[~mask_used]
        
        # Sort used vertices by first occurrence, then by index
        if len(used_verts) > 0:
            sort_keys = np.column_stack([first_occurrence[used_verts], used_verts])
            sorted_indices = np.lexsort((sort_keys[:, 1], sort_keys[:, 0]))
            used_verts = used_verts[sorted_indices]
        
        # Concatenate used then unused
        ordered_vertices = np.concatenate([used_verts, unused_verts])
        
        # Split into slices
        return np.array_split(ordered_vertices, num_slices)


    def embed_message(self, message: str, bits_per_coord=DEFAULT_BITS_PER_COORD, scale=DEFAULT_STEGO_SCALE):
        """
        Embed a text message directly into the vertex coordinates of this model.
        Uses index-based traversal that survives export/import cycles.
        """
        if self.positions is None or self.vertex_count == 0:
            print("[Model] No positions to embed into.")
            return False

        if not message:
            print("[Model] Empty message, nothing to embed.")
            return False

        if len(message) > 65535:
            print("[Model] Message too long (max 65535 characters).")
            return False

        # Store original positions before modification
        self.stego_original_positions = self.positions.copy()
        
        # Build bit string
        length_bits = f"{len(message):016b}"
        msg_bits = ''.join(f"{ord(c):08b}" for c in message)
        bin_msg = length_bits + msg_bits
        msg_len_bits = len(bin_msg)

        # Use index-based traversal (survives export/import)
        slice_vertex_indices = self._build_slice_vertex_indices_from_indices()
        total_vertices_for_stego = int(sum(len(arr) for arr in slice_vertex_indices))
        total_coords = total_vertices_for_stego * 3
        capacity_bits = total_coords * bits_per_coord
        
        if msg_len_bits > capacity_bits:
            print(f"[Model] Message too long. Capacity={capacity_bits} bits, needed={msg_len_bits}")
            return False

        print(f"[Model] Using index-based traversal: {total_vertices_for_stego} vertices, {capacity_bits} bit capacity")
        
        # For VTK files, use higher scale to preserve precision
        if hasattr(self, 'file_extension') and self.file_extension == '.vtk':
            effective_scale = scale * 10  # Higher scale for VTK
            print(f"[Model] Using enhanced scale {effective_scale} for VTK file")
        else:
            effective_scale = scale
        
        # Modify vertices
        verts = self.positions.copy()
        flat = verts.reshape(-1)
        bit_idx = 0

        for v_indices in slice_vertex_indices:
            if bit_idx >= msg_len_bits:
                break
            for vid in v_indices:
                if bit_idx >= msg_len_bits:
                    break
                base = int(vid) * 3
                for ci in range(3):
                    if bit_idx >= msg_len_bits:
                        break
                    coord_idx = base + ci
                    # Use effective scale
                    val = int(round(float(flat[coord_idx]) * effective_scale))
                    for b in range(bits_per_coord):
                        if bit_idx >= msg_len_bits:
                            break
                        bit = int(bin_msg[bit_idx])
                        val = (val & ~(1 << b)) | (bit << b)
                        bit_idx += 1
                    flat[coord_idx] = val / float(effective_scale)

        self.positions = flat.reshape((-1, 3)).astype(np.float32)

        # Track changed vertices
        try:
            diff = np.any(self.stego_original_positions != self.positions, axis=1)
            self.changed_vertex_mask = diff
            print(f"[Model] Stego modified {int(diff.sum()):,} vertices.")
        except Exception as e:
            print("[Model] Failed to compute changed_vertex_mask:", e)
            self.changed_vertex_mask = None

        # Rebuild normals and GPU buffers
        try:
            self.normals = compute_vertex_normals(self.positions, self.indices)
        except Exception as e:
            print("[Model] Failed to recompute normals:", e)
        
        self._build_chunks()
        self.slice_points = None
        return True

    def extract_message(self, bits_per_coord=DEFAULT_BITS_PER_COORD, scale=DEFAULT_STEGO_SCALE):
        """
        Extract a message using index-based traversal.
        Works even after export/import because it only depends on triangle indices.
        """
        if self.positions is None or self.vertex_count == 0:
            print("[Model] No positions to extract from.")
            return ""

        # Use index-based traversal (same as embedding)
        slice_vertex_indices = self._build_slice_vertex_indices_from_indices()
        print(f"[Model] Extracting using index-based traversal")
        
        # For VTK files, use higher scale
        if hasattr(self, 'file_extension') and self.file_extension == '.vtk':
            effective_scale = scale * 10  # Higher scale for VTK
            print(f"[Model] Using enhanced scale {effective_scale} for VTK extraction")
        else:
            effective_scale = scale
        
        verts = self.positions
        flat = verts.reshape(-1)
        
        all_bits = ''
        
        for v_indices in slice_vertex_indices:
            for vid in v_indices:
                base = int(vid) * 3
                for ci in range(3):
                    coord_idx = base + ci
                    val = int(round(float(flat[coord_idx]) * effective_scale))
                    for b in range(bits_per_coord):
                        bit = (val >> b) & 1
                        all_bits += str(bit)
        
        if len(all_bits) < 16:
            print(f"[Model] Not enough bits. Need 16, got {len(all_bits)}")
            return ""
        
        try:
            msg_len_bytes = int(all_bits[:16], 2)
        except Exception as e:
            print("[Model] Invalid length header:", e)
            return ""
        
        if msg_len_bytes <= 0 or msg_len_bytes > 65535:
            print(f"[Model] Invalid message length: {msg_len_bytes}")
            return ""
        
        total_message_bits = msg_len_bytes * 8
        total_needed_bits = 16 + total_message_bits
        
        if len(all_bits) < total_needed_bits:
            print(f"[Model] Not enough bits for message. Need {total_needed_bits}, got {len(all_bits)}")
            return ""
        
        data_bits = all_bits[16:16+total_message_bits]
        
        chars = []
        try:
            for i in range(0, len(data_bits), 8):
                byte = data_bits[i:i+8]
                if len(byte) == 8:
                    char_code = int(byte, 2)
                    chars.append(chr(char_code))
        except Exception as e:
            print("[Model] Failed decoding:", e)
            return ""
        
        result = ''.join(chars)
        print(f"[Model] Extracted message: {len(result)} characters")
        return result

    def _build_slice_vertex_indices(self, axis=None, num_slices=None, epsilon=None):
        """
        DEPRECATED: This method is kept for backward compatibility with slicing visualization.
        For steganography, use _build_slice_vertex_indices_from_indices() instead.
        """
        if self.positions is None or self.vertex_count == 0:
            return []

        axis = (axis or self.slice_axis).upper()
        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis, 2)

        scale = DEFAULT_STEGO_SCALE
        verts_int = np.round(self.positions * scale).astype(np.int64)

        other_axes = [i for i in (0, 1, 2) if i != axis_idx]

        order = np.lexsort((
            verts_int[:, other_axes[1]],
            verts_int[:, other_axes[0]],
            verts_int[:, axis_idx]
        ))

        num_slices = num_slices or self.slice_count
        ordered_indices = order.astype(np.int32)

        return np.array_split(ordered_indices, num_slices)

    def export_model(self, filename: str, original_ext: str):
        """
        Export the current model with modified vertices preserved.
        Supports STL, OBJ, and VTK, matching the original file format.
        """
        if self.positions is None or self.indices is None:
            raise ValueError("No model data to export.")

        ext = original_ext.lower()
        if ext == '.stl':
            self._export_stl(filename)
        elif ext == '.obj':
            self._export_obj(filename)
        elif ext == '.vtk':
            self._export_vtk(filename)
        else:
            raise ValueError("Unsupported export format.")

    def _export_obj(self, filename: str):
        """
        Export indexed mesh to OBJ using current (possibly stego-modified) vertices.
        """
        with open(filename, 'w') as f:
            f.write("# Exported with steganographic vertices\n")

            # Write vertices
            for v in self.positions:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Write faces (OBJ is 1-based)
            tris = self.indices.reshape(-1, 3)
            for a, b, c in tris:
                f.write(f"f {a+1} {b+1} {c+1}\n")

    def _export_stl(self, filename: str):
        """
        Export binary STL with modified vertex positions preserved.
        """
        tris = self.indices.reshape(-1, 3)

        with open(filename, 'wb') as f:
            header = b"Stego STL Export".ljust(80, b'\0')
            f.write(header)
            f.write(struct.pack('<I', len(tris)))

            for a, b, c in tris:
                v1 = self.positions[a]
                v2 = self.positions[b]
                v3 = self.positions[c]

                # Compute face normal
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal /= norm
                else:
                    normal[:] = 0.0

                f.write(struct.pack('<3f', *normal))
                f.write(struct.pack('<3f', *v1))
                f.write(struct.pack('<3f', *v2))
                f.write(struct.pack('<3f', *v3))
                f.write(struct.pack('<H', 0))  # attribute byte count

    def _export_vtk(self, filename: str):
        """
        Export indexed mesh to VTK using current (possibly stego-modified) vertices.
        Preserves the exact vertex positions and triangle connectivity.
        """
        if not VTK_AVAILABLE:
            raise ImportError("VTK module not available. Cannot export VTK.")
        
        # Create a vtkPolyData object
        poly = vtk.vtkPolyData()
        
        # Create points - use DOUBLE precision for VTK to preserve stego bits
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()  # Use double precision
        points.SetNumberOfPoints(self.vertex_count)
        
        for i in range(self.vertex_count):
            # Use double precision coordinates
            points.SetPoint(i, 
                float(self.positions[i][0]),
                float(self.positions[i][1]), 
                float(self.positions[i][2])
            )
        
        poly.SetPoints(points)
        
        # Create cells (triangles) - preserve exact order
        cells = vtk.vtkCellArray()
        tris = self.indices.reshape(-1, 3)
        
        for tri in tris:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, int(tri[0]))
            triangle.GetPointIds().SetId(1, int(tri[1]))
            triangle.GetPointIds().SetId(2, int(tri[2]))
            cells.InsertNextCell(triangle)
        
        poly.SetPolys(cells)
        
        # Write the file with maximum precision
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(poly)
        writer.SetFileTypeToBinary()
        writer.Write()
        
        print(f"[Model] Exported VTK with double precision: {self.vertex_count} vertices, {len(tris)} triangles")

    def set_slice_axis(self, axis: str):
        """
        Set the slice axis and recompute slices.
        axis: 'X', 'Y', or 'Z'
        """
        axis = axis.upper()
        if axis not in ('X', 'Y', 'Z'):
            axis = 'Z'
        self.slice_axis = axis
        # Recompute slices with new axis
        self.compute_slices(axis=self.slice_axis, num_slices=self.slice_count)
    # -------------------------------------------------
    # Slicing utilities
    # -------------------------------------------------
    def compute_slices(self, axis=DEFAULT_SLICE_AXIS, num_slices=DEFAULT_NUM_SLICES, epsilon=1e-3):
        """
        Compute 2D slices of the vertex cloud along the given axis.
        Stores slice_points as a list of (N_i, 2) arrays in model space.
        """
        if self.positions is None or self.vertex_count == 0:
            self.slice_points = []
            self.slice_axis = axis
            self.slice_count = num_slices
            return

        axis = axis.upper()
        axis_idx_map = {'X': 0, 'Y': 1, 'Z': 2}
        if axis not in axis_idx_map:
            axis = 'Z'
        axis_idx = axis_idx_map[axis]
        coords = self.positions
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)

        planes = np.linspace(mins[axis_idx], maxs[axis_idx], num_slices)
        slices = []
        slices3d = []

        # choose which components form 2D slice (drop slicing axis)
        if axis_idx == 0:
            other_axes = (1, 2)
        elif axis_idx == 1:
            other_axes = (0, 2)
        else:
            other_axes = (0, 1)

        for plane in planes:
            mask = np.abs(coords[:, axis_idx] - plane) < epsilon
            pts3d = coords[mask]
            pts2d = pts3d[:, other_axes] if pts3d.size > 0 else np.zeros((0, 2), dtype=np.float32)
            slices.append(pts2d.astype(np.float32))
            slices3d.append(pts3d.astype(np.float32))

        self.slice_points = slices
        self.slice_points_3d = slices3d
        self.slice_axis = axis
        self.slice_count = num_slices
        self.slice_planes = planes
        self.slice_bounds = (mins, maxs)

        print(f"[Model] Unique verts: {self.vertex_count:,}, indexed triangles: {self.triangle_count:,}")

        # now chunk indices into groups of TRIANGLES_PER_CHUNK triangles
        tris_total = self.triangle_count
        tris_per_chunk = TRIANGLES_PER_CHUNK
        if tris_per_chunk <= 0:
            tris_per_chunk = tris_total

        # Each chunk will reference a subset of vertices -> we must extract per-chunk local vertex arrays
        # We'll go chunk-by-chunk and upload independent VBO/NBO/EBO for each chunk
        self.chunks = []

        tri_indices = self.indices.reshape(-1, 3)
        start_tri = 0
        try:
            while start_tri < tris_total:
                end_tri = min(tris_total, start_tri + tris_per_chunk)
                chunk_tris = tri_indices[start_tri:end_tri]  # shape (K,3)
                # find unique vertex indices referenced by this chunk and build local arrays
                unique_idx, inv = np.unique(chunk_tris.flatten(), return_inverse=True)
                # positions subset
                pos_chunk = self.positions[unique_idx]
                norm_chunk = self.normals[unique_idx]
                # remap chunk indices to local
                local_indices = inv.astype(np.uint32).reshape(-1)
                # create chunk and upload
                chunk = MeshChunk()
                chunk.upload(pos_chunk.astype(np.float32), norm_chunk.astype(np.float32), local_indices)
                self.chunks.append(chunk)
                start_tri = end_tri
        except Exception as e:
            print("[Model] GPU upload failed - falling back to streaming mode:", e)
            # Clean up partial uploads
            for c in self.chunks:
                try: c.release()
                except: pass
            self.chunks = []
            self.streaming_mode = True
            # streaming_mode -> we'll upload each frame as one VBO+NBO+EBO and draw then delete (slower but safe)

    def draw(self):
        if self.streaming_mode:
            # stream upload per-frame (safe but slower) - upload whole mesh each frame then draw and delete
            try:
                # single chunk streaming
                chunk = MeshChunk()
                indices_uint32 = self.indices.astype(np.uint32)
                chunk.upload(self.positions.astype(np.float32), self.normals.astype(np.float32), indices_uint32)
                chunk.draw()
                chunk.release()
            except Exception as e:
                print("[Model] Streaming draw failed:", e)
            return
        # normal case: draw all chunks
        for chunk in self.chunks:
            try:
                chunk.draw()
            except Exception as e:
                print("[Model] Failed draw chunk:", e)

    def release(self):
        self.clear_gpu()
        self.positions = None
        self.indices = None
        self.normals = None
        self.changed_vertex_mask = None
        self.streaming_mode = False
        self.vertex_count = 0
        self.triangle_count = 0

# -------------------------
# Basic cube fallback (visual)
# -------------------------
cube_vertices = np.array([
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1, -1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1, -1,  1],
    [-1,  1,  1],
], dtype=np.float32)

cube_surfaces = (
    (0,1,2,3),
    (4,5,7,6),
    (0,4,6,3),
    (1,5,7,2),
    (0,1,5,4),
    (3,2,7,6)
)
cube_colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (1,0,1),
    (0,1,1)
)

def draw_cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(cube_surfaces):
        glColor3fv(cube_colors[i])
        for vi in face:
            glVertex3fv(cube_vertices[vi])
    glEnd()

# -------------------------
# Viewport / init
# -------------------------
def resize_viewport(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.01, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init_display(width, height):
    pygame.init()
    # Initialize the Pygame font module
    if not pygame.font.get_init():
        pygame.font.init()
        
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    resize_viewport(width, height)

# -------------------------
# Viewer Application
# -------------------------
class Viewer:
    def __init__(self, width=DEFAULT_WINDOW[0], height=DEFAULT_WINDOW[1]):
        self.width = width
        self.height = height
        init_display(self.width, self.height)

        self.zoom_z = -3.5
        self.rotation_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

        self.is_dragging = False
        self.last_x = 0
        self.last_y = 0

        self.model = None
        self.clock = pygame.time.Clock() # Pygame clock for FPS
        self.stats_font = pygame.font.Font(None, FPS_FONT_SIZE) # Font object
        self.show_vertices = False          # Toggle for all vertex display (black dots)
        self.show_changed_vertices = False  # Toggle for stego-modified vertices (red dots)

        # Slicing / steganography UI state
        self.slice_mode = False
        self.current_slice_index = 0
        self.slice_surfaces = []  # cached 2D slice images
        self.slice_view_2d = False  # Whether we're in 2D slice view mode (replaces 3D view)
        self.slice_grid_cols = 8  # Number of columns in slice grid
        self.back_to_3d_button_rect = None  # Rectangle for "Back to 3D" button

        # Toolbar configuration
        self.toolbar_height = 50
        self.button_font = pygame.font.Font(None, 24)
        self.button_height = 40
        self.button_padding = 5
        self.button_spacing = 5
        self.toolbar_y = self.height - self.toolbar_height
        self.hovered_button = None  # Track which button is hovered
        self.mouse_pos = (0, 0)  # Track mouse position
        
        # Adjust viewport to exclude toolbar area
        viewport_height = self.height - self.toolbar_height
        resize_viewport(self.width, viewport_height)

        # lighting
        glEnable(GL_LIGHTING)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0.6, 1.0, 0.8, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  (0.95, 0.95, 0.95, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, (-0.7, -0.3, 0.5, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  (0.5, 0.5, 0.5, 1.0))

        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.25, 0.25, 0.25, 1.0))

        # material defaults
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (0.9,0.9,0.9,1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.2,0.2,0.2,1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 8.0)

        self.slice_epsilon = 1e-3

        # run loop
        self.main_loop()

    def draw_progress_bar(self, progress, message="Loading..."):
        """Draw a progress bar just above the toolbar"""
        bar_width = 800
        bar_height = 15
        bar_x = (self.width - bar_width) // 2
        bar_y = self.toolbar_y - bar_height
        
        # Create surface for progress bar
        progress_surf = pygame.Surface((self.width, bar_height + 40), pygame.SRCALPHA)
        
        # Background
        bg_rect = pygame.Rect(bar_x, 0, bar_width, bar_height)
        pygame.draw.rect(progress_surf, (30, 30, 35, 230), bg_rect)
        pygame.draw.rect(progress_surf, (80, 80, 90, 255), bg_rect, 2)
        
        # Progress fill
        fill_width = int((bar_width - 4) * progress)
        if fill_width > 0:
            fill_rect = pygame.Rect(bar_x + 2, 2, fill_width, bar_height - 4)
            # Gradient fill
            for x in range(fill_width):
                gradient_factor = 0.7 + (x / max(fill_width, 1)) * 0.3
                r = int(70 * gradient_factor)
                g = int(130 * gradient_factor)
                b = int(180 * gradient_factor)
                pygame.draw.line(progress_surf, (r, g, b, 255),
                            (bar_x + 2 + x, 2),
                            (bar_x + 2 + x, bar_height - 2))
        
        # Text
        font = pygame.font.Font(None, 22)
        text_surf = font.render(f" ", True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(self.width // 2, bar_height + 10))
        progress_surf.blit(text_surf, text_rect)
        
        # Render to OpenGL
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        pixels = pygame.image.tostring(progress_surf, "RGBA", True)
        glWindowPos2f(0, bar_y - 40)
        glDrawPixels(self.width, bar_height + 40, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        glPopAttrib()
        pygame.display.flip()

    def import_model_dialog(self):
        root = tk.Tk()
        root.withdraw()
        # Update filetypes to include VTK
        filename = filedialog.askopenfilename(
            title="Import 3D Model",
            filetypes=[
                ("3D files", "*.stl *.obj *.vtk"), 
                ("STL", "*.stl"), 
                ("OBJ", "*.obj"),
                ("VTK", "*.vtk")
            ]
        )
        root.destroy()
        if not filename:
            return

        # free old
        if self.model:
            try:
                self.model.release()
            except:
                pass
            self.model = None

        try:
            # Show initial progress
            self.draw_progress_bar(0.0, "Loading model")
            
            # Create new model and load it
            self.model = Model()
            
            # Load file using Model.load_from_file()
            self.draw_progress_bar(0.3, "Reading file")
            self.model.load_from_file(filename)
            
            self.draw_progress_bar(0.9, "Processing geometry")
            
            self.draw_progress_bar(1.0, "Complete")
            
            print(f"[Viewer] Model loaded: triangles={self.model.triangle_count:,}, unique vertices={self.model.vertex_count:,}")
            
            # Only compute slices for surface meshes
            if self.model.triangle_count > 0:
                # Recompute slices and surfaces
                self.model.compute_slices()
                self.build_slice_surfaces()
                
        except ImportError as e:
            print("[Viewer] VTK module not available:", e)
            self._tk_info("VTK Error", "VTK module not available. Please install vtk package:\npip install vtk")
        except Exception as e:
            print("[Viewer] Failed to load model:", e)
            import traceback
            traceback.print_exc()
            self.model = None  # Reset model if loading failed

    def export_model_dialog(self):
        if not self.model:
            self._tk_info("Export", "No model loaded.")
            return

        root = tk.Tk()
        root.withdraw()

        ext = self.model.original_ext
        default_name = f"stego_export{ext}"

        # Include VTK in export options if it was a VTK file originally
        if ext == '.vtk':
            filetypes = [("VTK", "*.vtk"), ("OBJ", "*.obj"), ("STL", "*.stl")]
        else:
            filetypes = [("OBJ", "*.obj"), ("STL", "*.stl")]
        
        filename = filedialog.asksaveasfilename(
            title="Export Stego Model",
            defaultextension=ext,
            initialfile=default_name,
            filetypes=filetypes
        )

        root.destroy()

        if not filename:
            return

        # Get the actual extension from the chosen filename
        chosen_ext = os.path.splitext(filename)[1].lower()
        if not chosen_ext:
            chosen_ext = ext  # Use original extension if none provided
        
        try:
            self.model.export_model(filename, chosen_ext)
            self._tk_info("Export", f"Model exported successfully.\nSteganographic data preserved in {chosen_ext.upper()} format.")
        except ImportError as e:
            self._tk_info("VTK Export Error", "VTK module not available for export.\nPlease install vtk package.")
        except Exception as e:
            self._tk_info("Export Failed", str(e))


    def embed_with_progress(self, message):
        """Embed message with progress bar"""
        if not self.model or not message:
            return False
        
        self.draw_progress_bar(0.0, "Embedding")
        
        # The actual embedding
        result = self.model.embed_message(message)
        
        self.draw_progress_bar(1.0, "Complete")
        return result

    def recompute_slices_with_progress(self):
        """Recompute slices with progress bar"""
        self.draw_progress_bar(0.0, "Computing slices")
        
        self.model.compute_slices(axis=self.model.slice_axis,
                                num_slices=self.model.slice_count,
                                epsilon=self.slice_epsilon)
        
        self.draw_progress_bar(0.7, "Building surfaces")
        self.build_slice_surfaces()
        
        self.draw_progress_bar(1.0, "Complete")

    def _tk_text_input(self, title, prompt, initial=""):
        """Blocking Tkinter dialog to get text input (GUI, no console)."""
        root = tk.Tk()
        root.withdraw()
        try:
            value = simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)
        except Exception:
            value = None
        root.destroy()
        return value

    def _tk_info(self, title, message):
        """Small Tkinter info box."""
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showinfo(title, message, parent=root)
        except Exception:
            pass
        root.destroy()

    def build_slice_surfaces(self, size=512):
        """
        Convert model.slice_points into simple 2D images (pygame surfaces).
        Changed vertices are red, unchanged are green. Uses smaller pixel points for better accuracy.
        """
        self.slice_surfaces = []
        # Check if we have a model
        if not self.model:
            return
        

        # Compute global bounds across all slices for consistent scaling
        all_pts = [s for s in self.model.slice_points if s is not None and s.size > 0]
        if not all_pts:
            return
        all_pts = np.vstack(all_pts)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = maxs - mins
        span[span == 0] = 1.0  # avoid div by zero

        # Build slice vertex indices for coloring
        slice_vertex_indices = self.model._build_slice_vertex_indices()

        for slice_idx, pts in enumerate(self.model.slice_points):
            surf = pygame.Surface((size, size), pygame.SRCALPHA)
            # Fill with dark background for visibility
            surf.fill((15, 15, 20, 255))
            
            if pts is not None and pts.size > 0:
                # normalize to [0,1]
                norm = (pts - mins) / span
                # flip Y so it looks natural
                xs = (norm[:, 0] * (size - 8) + 4).astype(int)
                ys = ((1.0 - norm[:, 1]) * (size - 8) + 4).astype(int)
                
                # Get vertex indices for this slice
                if slice_idx < len(slice_vertex_indices):
                    v_indices = slice_vertex_indices[slice_idx]
                    
                    # Determine color for each vertex: red if changed, green if unchanged
                    # Use smaller single-pixel dots for more accurate visualization
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        if i < len(v_indices):
                            vid = v_indices[i]
                            # Check if this vertex was changed by steganography
                            if (self.model.changed_vertex_mask is not None and 
                                vid < len(self.model.changed_vertex_mask) and 
                                self.model.changed_vertex_mask[vid]):
                                color = (255, 0, 0, 255)  # Red for changed
                            else:
                                color = (0, 255, 100, 255)  # Green for unchanged
                        else:
                            color = (0, 255, 100, 255)  # Default green
                        
                        # Draw single pixel for maximum accuracy
                        if 0 <= x < size and 0 <= y < size:
                            surf.set_at((int(x), int(y)), color)
                else:
                    # Fallback if indices not available
                    for x, y in zip(xs, ys):
                        if 0 <= x < size and 0 <= y < size:
                            surf.set_at((int(x), int(y)), (0, 255, 100, 255))
            
            self.slice_surfaces.append(surf)

        # clamp current index
        if self.current_slice_index >= len(self.slice_surfaces):
            self.current_slice_index = max(0, len(self.slice_surfaces) - 1)

    def draw_stats_overlay(self):
        """Draw FPS, vertices, and faces counter at top-left"""
        fps = self.clock.get_fps()
        
        # Get model stats or use defaults (cube stats)
        if self.model and self.model.triangle_count > 0:
            vertices = self.model.vertex_count
            faces = self.model.triangle_count
        else:
            vertices = 8  # cube vertices
            faces = 12    # cube faces (6 quads = 12 triangles)
        
        # Create text lines
        lines = [
            f"FPS: {fps:.1f}",
            f"Vertices: {vertices:,}",
            f"Faces: {faces:,}"
        ]
        
        # Save current OpenGL state
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        margin = 10
        line_height = FPS_FONT_SIZE + 2
        y_offset = self.height - margin
        
        # Render each line
        for line in lines:
            # Render the text to a Pygame surface
            text_surface = self.stats_font.render(line, True, (255, 255, 255, 255))
            
            # Get dimensions and pixel data
            text_width = text_surface.get_width()
            text_height = text_surface.get_height()
            
            # Convert surface to string for OpenGL texture loading
            text_data = pygame.image.tostring(text_surface, "RGBA", True)

            # Position at current y_offset
            y_offset -= text_height
            glWindowPos2f(margin, y_offset)

            # Draw the pixels
            glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            
            # Move to next line
            y_offset -= 2  # small gap between lines

        # Restore OpenGL state
        glPopAttrib()

    def handle_events(self):
        """Modified event handler - restored keyboard navigation for slices"""
        for event in pygame.event.get():
            if event.type == QUIT:
                if self.model:
                    self.model.release()
                pygame.quit()
                sys.exit(0)

            elif event.type == VIDEORESIZE:
                self.width, self.height = event.size
                self.toolbar_y = self.height - self.toolbar_height
                viewport_height = self.height - self.toolbar_height
                resize_viewport(self.width, viewport_height)

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    if event.pos[1] >= self.toolbar_y:
                        self.handle_toolbar_click(event.pos)
                    else:
                        self.is_dragging = True
                        self.last_x, self.last_y = event.pos
                elif event.button == 4:
                    zoom_factor = 0.9
                    self.zoom_z *= zoom_factor
                elif event.button == 5:
                    zoom_factor = 1.1
                    self.zoom_z *= zoom_factor

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False
                    if event.pos[1] < self.toolbar_y:
                        self.hovered_button = None
                    else:
                        self.update_toolbar_hover(event.pos)

            elif event.type == MOUSEMOTION:
                self.mouse_pos = event.pos
                if event.pos[1] >= self.toolbar_y:
                    self.update_toolbar_hover(event.pos)
                else:
                    self.hovered_button = None
                
                if self.is_dragging:
                    dx = event.pos[0] - self.last_x
                    dy = event.pos[1] - self.last_y

                    glLoadIdentity()
                    glRotatef(dy * 0.5, 1, 0, 0)
                    glRotatef(dx * 0.5, 0, 1, 0)
                    glMultMatrixf(self.rotation_matrix)
                    self.rotation_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)

                    self.last_x, self.last_y = event.pos

            elif event.type == KEYDOWN:
                if event.key == pygame.K_i:
                    self.import_model_dialog()
                elif event.key == pygame.K_v:
                    self.show_vertices = not self.show_vertices
                    print(f"[Viewer] Vertex display: {'ON' if self.show_vertices else 'OFF'}")
                elif event.key == pygame.K_c:
                    self.show_changed_vertices = not self.show_changed_vertices
                    print(f"[Viewer] Changed-vertex display: {'ON' if self.show_changed_vertices else 'OFF'}")
                elif event.key == pygame.K_e:
                    if self.model:
                        msg = self._tk_text_input("Embed Message", "Enter message to embed into mesh:")
                        if msg is not None and msg != "":
                            ok = self.embed_with_progress(msg)
                            if ok:
                                self._tk_info("Embed Message", "Message embedded into mesh.")
                                if self.slice_mode:
                                    self.recompute_slices_with_progress()
                            else:
                                self._tk_info("Embed Message", "Failed to embed message (maybe too long?).")
                    else:
                        self._tk_info("Embed Message", "No model loaded.")
                elif event.key == pygame.K_x:
                    if self.model:
                        text = self.model.extract_message()
                        if text:
                            self._tk_info("Extracted Message", text)
                        else:
                            self._tk_info("Extract Message", "No valid message could be extracted.")
                    else:
                        self._tk_info("Extract Message", "No model loaded.")
                elif event.key == pygame.K_s:
                    if self.model and self.model.changed_vertex_mask is not None and np.any(self.model.changed_vertex_mask):
                        self.model.compute_slices(axis=self.model.slice_axis,
                                                num_slices=self.model.slice_count)
                        self.build_slice_surfaces()
                        self.slice_mode = not self.slice_mode
                        if self.slice_mode:
                            self.slice_view_2d = True
                        else:
                            self.slice_view_2d = False
                    elif self.model:
                        self._tk_info("Slices", "Slices are available only after embedding a message (press E first).")
                    else:
                        self._tk_info("Slices", "No model loaded.")
                # Restore LEFT and RIGHT arrow key handlers for slice navigation
                elif event.key == pygame.K_LEFT:
                    if self.slice_mode and self.slice_surfaces:
                        self.current_slice_index = (self.current_slice_index - 1) % len(self.slice_surfaces)
                elif event.key == pygame.K_RIGHT:
                    if self.slice_mode and self.slice_surfaces:
                        self.current_slice_index = (self.current_slice_index + 1) % len(self.slice_surfaces)

    def is_button_disabled(self, action_key):
        """Check if a button should be disabled based on current state"""
        if action_key == 'i':
            return False
        elif action_key == 'v':
            return self.model is None
        elif action_key == 'c':
            return (self.model is None or 
                self.model.changed_vertex_mask is None or 
                not np.any(self.model.changed_vertex_mask))
        elif action_key == 'e':
            return self.model is None
        elif action_key == 'x':
            return self.model is None
        elif action_key == 'export':
            return self.model is None
        elif action_key == 's':
            return (self.model is None or 
                self.model.changed_vertex_mask is None or 
                not np.any(self.model.changed_vertex_mask))
        elif action_key == 'config':
            return self.model is None
        elif action_key == 'axis':
            return self.model is None
        return False

    def update_toolbar_hover(self, pos):
        """Update which button is being hovered"""
        x, y = pos
        if y < self.toolbar_y:
            self.hovered_button = None
            return
        
        toolbar_relative_y = y - self.toolbar_y
        button_x = self.button_padding
        button_y = (self.toolbar_height - self.button_height) // 2
        
        # Added Config button
        buttons = [
            ("Import (I)", 100, 'i'),
            ("Vertices (V)", 120, 'v'),
            ("Changed (C)", 120, 'c'),
            ("Embed (E)", 100, 'e'),
            ("Extract (X)", 110, 'x'),
            ("Export", 110, 'export'),
            ("Slices (S)", 100, 's'),
            ("Axis", 80, 'axis'),  # NEW BUTTON
            ("Config", 90, 'config'),
        ]
        
        self.hovered_button = None
        for label, width, action_key in buttons:
            if button_x <= x <= button_x + width and button_y <= toolbar_relative_y <= button_y + self.button_height:
                if not self.is_button_disabled(action_key):
                    self.hovered_button = action_key
                break
            button_x += width + self.button_spacing

    def handle_toolbar_click(self, pos):
        """Handle mouse clicks on toolbar buttons"""
        x, y = pos
        if y < self.toolbar_y:
            return
        
        toolbar_relative_y = y - self.toolbar_y
        button_x = self.button_padding
        button_y = (self.toolbar_height - self.button_height) // 2
        
        buttons = [
            ("Import (I)", 100, 'i'),
            ("Vertices (V)", 120, 'v'),
            ("Changed (C)", 120, 'c'),
            ("Embed (E)", 100, 'e'),
            ("Extract (X)", 110, 'x'),
            ("Export", 110, 'export'),
            ("Slices (S)", 100, 's'),
            ("Axis", 80, 'axis'),  # NEW BUTTON
            ("Config", 90, 'config'),
        ]
        
        for label, width, action_key in buttons:
            if button_x <= x <= button_x + width and button_y <= toolbar_relative_y <= button_y + self.button_height:
                if self.is_button_disabled(action_key):
                    break
                
                if action_key == 'i':
                    self.import_model_dialog()
                elif action_key == 'v':
                    self.show_vertices = not self.show_vertices
                    print(f"[Viewer] Vertex display: {'ON' if self.show_vertices else 'OFF'}")
                elif action_key == 'c':
                    self.show_changed_vertices = not self.show_changed_vertices
                    print(f"[Viewer] Changed-vertex display: {'ON' if self.show_changed_vertices else 'OFF'}")
                elif action_key == 'e':
                    msg = self._tk_text_input("Embed Message", "Enter message to embed into mesh:")
                    if msg is not None and msg != "":
                        ok = self.embed_with_progress(msg)
                        if ok:
                            self._tk_info("Embed Message", "Message embedded into mesh.")
                            if self.slice_mode:
                                self.recompute_slices_with_progress()
                        else:
                            self._tk_info("Embed Message", "Failed to embed message (maybe too long?).")
                elif action_key == 'x':
                    text = self.model.extract_message()
                    if text:
                        self._tk_info("Extracted Message", text)
                    else:
                        self._tk_info("Extract Message", "No valid message could be extracted.")
                elif action_key == 'export':
                    self.export_model_dialog()
                elif action_key == 's':
                    self.model.compute_slices(axis=self.model.slice_axis,
                                            num_slices=self.model.slice_count,
                                            epsilon=self.slice_epsilon)
                    self.build_slice_surfaces()
                    self.slice_mode = not self.slice_mode
                    if self.slice_mode:
                        self.slice_view_2d = True
                    else:
                        self.slice_view_2d = False
                elif action_key == 'config':
                    self.show_config_dialog()
                elif action_key == 'axis':
                    self.show_axis_selection_dialog()
                break
            button_x += width + self.button_spacing

    def show_config_dialog(self):
        """Show configuration dialog for slice parameters including axis selection"""
        root = tk.Tk()
        root.withdraw()
        
        # Get slice axis
        axis_result = simpledialog.askstring(
            "Slice Configuration",
            f"Slice axis (X/Y/Z, current: {self.model.slice_axis}):",
            initialvalue=str(self.model.slice_axis),
            parent=root
        )
        
        if axis_result is not None:
            axis_result = axis_result.upper()
            if axis_result in ('X', 'Y', 'Z'):
                self.model.slice_axis = axis_result
        
        # Get number of slices
        num_slices_str = simpledialog.askstring(
            "Slice Configuration",
            f"Number of slices (current: {self.model.slice_count}):",
            initialvalue=str(self.model.slice_count),
            parent=root
        )
        
        if num_slices_str is not None:
            try:
                num_slices = int(num_slices_str)
                if num_slices < 1:
                    num_slices = 1
                if num_slices > 1000:
                    num_slices = 1000
                self.model.slice_count = num_slices
            except ValueError:
                pass
        
        # Get epsilon
        epsilon_str = simpledialog.askstring(
            "Slice Configuration",
            f"Epsilon (vertex tolerance, current: {self.slice_epsilon}):",
            initialvalue=str(self.slice_epsilon),
            parent=root
        )
        
        if epsilon_str is not None:
            try:
                epsilon = float(epsilon_str)
                if epsilon <= 0:
                    epsilon = 1e-6
                self.slice_epsilon = epsilon
            except ValueError:
                pass
        
        root.destroy()
        
        # Recompute slices with new parameters if in slice mode
        if self.slice_mode and self.model:
            self.draw_progress_bar(0.0, "Reconfiguring")
            self.model.compute_slices(axis=self.model.slice_axis,
                                    num_slices=self.model.slice_count,
                                    epsilon=self.slice_epsilon)
            self.draw_progress_bar(0.7, "Building surfaces")
            self.build_slice_surfaces()
            self.draw_progress_bar(1.0, "Complete")
            self._tk_info("Configuration", f"Updated: axis={self.model.slice_axis}, {self.model.slice_count} slices, epsilon={self.slice_epsilon}")

    def show_axis_selection_dialog(self):
        """Show dialog to select slice axis only"""
        root = tk.Tk()
        root.withdraw()
        
        axis_result = simpledialog.askstring(
            "Select Slice Axis",
            "Enter slice axis (X, Y, or Z):",
            initialvalue=str(self.model.slice_axis),
            parent=root
        )
        
        if axis_result is not None:
            axis_result = axis_result.upper()
            if axis_result in ('X', 'Y', 'Z'):
                # Update model slice axis
                self.model.slice_axis = axis_result
                
                # Recompute slices if needed
                if self.slice_mode and self.model:
                    self.draw_progress_bar(0.0, f"Recomputing slices along {axis_result} axis")
                    self.model.compute_slices(axis=self.model.slice_axis,
                                            num_slices=self.model.slice_count,
                                            epsilon=self.slice_epsilon)
                    self.draw_progress_bar(0.7, "Building surfaces")
                    self.build_slice_surfaces()
                    self.draw_progress_bar(1.0, "Complete")
                    
                self._tk_info("Slice Axis", f"Updated slice axis to {axis_result}")
        
        root.destroy()

    def draw_toolbar(self):
        """Draw the toolbar panel at the bottom of the window with enhanced button design"""
        toolbar_surf = pygame.Surface((self.width, self.toolbar_height), pygame.SRCALPHA)
        
        for y in range(self.toolbar_height):
            alpha = int(255 * 0.95)
            color_factor = 0.85 + (y / self.toolbar_height) * 0.15
            r = int(35 * color_factor)
            g = int(35 * color_factor)
            b = int(40 * color_factor)
            pygame.draw.line(toolbar_surf, (r, g, b, alpha), (0, y), (self.width, y))
        
        pygame.draw.line(toolbar_surf, (80, 80, 90, 255), (0, 0), (self.width, 0), 1)
        pygame.draw.line(toolbar_surf, (60, 60, 70, 200), (0, 1), (self.width, 1), 1)
        
        # Added Config button
        buttons = [
            ("Import (I)", 100, 'i'),
            ("Vertices (V)", 120, 'v'),
            ("Changed (C)", 120, 'c'),
            ("Embed (E)", 100, 'e'),
            ("Extract (X)", 110, 'x'),
            ("Export", 110, 'export'),
            ("Slices (S)", 100, 's'),
            ("Axis", 80, 'axis'),  # NEW BUTTON
            ("Config", 90, 'config')
        ]
        
        button_x = self.button_padding
        button_y = (self.toolbar_height - self.button_height) // 2
        
        for label, width, action_key in buttons:
            button_rect = pygame.Rect(button_x, button_y, width, self.button_height)
            is_disabled = self.is_button_disabled(action_key)
            is_hovered = (self.hovered_button == action_key) and not is_disabled
            
            is_active = False
            if action_key == 'v' and self.show_vertices:
                is_active = True
            elif action_key == 'c' and self.show_changed_vertices:
                is_active = True
            elif action_key == 's' and self.slice_mode:
                is_active = True
            
            if is_disabled:
                base_color = (40, 40, 45)
                border_color = (50, 50, 55)
                text_color = (120, 120, 120)
                alpha = 150
            elif is_active:
                base_color = (70, 130, 180)
                border_color = (100, 170, 220)
                text_color = (200, 240, 255)
                alpha = 255
            elif is_hovered:
                base_color = (90, 90, 100)
                border_color = (140, 140, 150)
                text_color = (255, 255, 240)
                alpha = 255
            else:
                base_color = (65, 65, 75)
                border_color = (100, 100, 110)
                text_color = (255, 255, 255)
                alpha = 255
            
            shadow_rect = pygame.Rect(button_x + 2, button_y + 2, width, self.button_height)
            shadow_alpha = 50 if is_disabled else 100
            pygame.draw.rect(toolbar_surf, (0, 0, 0, shadow_alpha), shadow_rect, border_radius=4)
            
            for by in range(self.button_height):
                gradient_factor = 0.7 + (by / self.button_height) * 0.3
                r = int(base_color[0] * gradient_factor)
                g = int(base_color[1] * gradient_factor)
                b = int(base_color[2] * gradient_factor)
                if is_hovered and not is_disabled:
                    r = min(255, int(r * 1.15))
                    g = min(255, int(g * 1.15))
                    b = min(255, int(b * 1.15))
                final_alpha = alpha if is_disabled else 255
                pygame.draw.line(toolbar_surf, (r, g, b, final_alpha), 
                                (button_x, button_y + by), 
                                (button_x + width, button_y + by))
            
            border_width = 2 if (is_hovered or is_active) and not is_disabled else 1
            pygame.draw.line(toolbar_surf, border_color, 
                        (button_x + 2, button_y), 
                        (button_x + width - 2, button_y), border_width)
            pygame.draw.line(toolbar_surf, border_color, 
                        (button_x + 2, button_y + self.button_height - 1), 
                        (button_x + width - 2, button_y + self.button_height - 1), border_width)
            pygame.draw.line(toolbar_surf, border_color, 
                        (button_x, button_y + 2), 
                        (button_x, button_y + self.button_height - 2), border_width)
            pygame.draw.line(toolbar_surf, border_color, 
                        (button_x + width - 1, button_y + 2), 
                        (button_x + width - 1, button_y + self.button_height - 2), border_width)
            
            corner_radius = 3
            for corner in [(button_x + corner_radius, button_y + corner_radius),
                        (button_x + width - corner_radius, button_y + corner_radius),
                        (button_x + corner_radius, button_y + self.button_height - corner_radius),
                        (button_x + width - corner_radius, button_y + self.button_height - corner_radius)]:
                pygame.draw.circle(toolbar_surf, border_color, corner, corner_radius, border_width)
            
            shadow_alpha = 100 if is_disabled else 180
            shadow_surface = self.button_font.render(label, True, (0, 0, 0, shadow_alpha))
            shadow_rect = shadow_surface.get_rect(center=(button_x + width // 2 + 1, button_y + self.button_height // 2 + 1))
            toolbar_surf.blit(shadow_surface, shadow_rect)
            
            text_surface = self.button_font.render(label, True, text_color)
            text_rect = text_surface.get_rect(center=(button_x + width // 2, button_y + self.button_height // 2))
            toolbar_surf.blit(text_surface, text_rect)
            
            button_x += width + self.button_spacing
        
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        pixels = pygame.image.tostring(toolbar_surf, "RGBA", True)
        glWindowPos2f(0, 0)
        glDrawPixels(self.width, self.toolbar_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        glPopAttrib()

    def draw_vertices(self):
        """Draw black dots at each vertex position"""
        if not self.model or not self.model.positions.size:
            return
        
        # Save current state
        glPushAttrib(GL_LIGHTING_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        
        # Disable lighting for point rendering
        glDisable(GL_LIGHTING)
        
        # Set point size and color
        glPointSize(4.0)
        glColor3f(0.0, 0.0, 0.0)  # Black
        
        # Draw points
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.model.positions)
        glDrawArrays(GL_POINTS, 0, self.model.vertex_count)
        glDisableClientState(GL_VERTEX_ARRAY)
        
        # Restore state
        glPopAttrib()

    def draw_changed_vertices(self):
        """Draw red dots at vertices whose positions changed due to steganography."""
        if (not self.model or
            self.model.positions is None or
            self.model.changed_vertex_mask is None or
            not np.any(self.model.changed_vertex_mask)):
            return

        changed_positions = self.model.positions[self.model.changed_vertex_mask]
        count = changed_positions.shape[0]
        if count == 0:
            return

        glPushAttrib(GL_LIGHTING_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        glDisable(GL_LIGHTING)

        glPointSize(5.0)
        glColor3f(1.0, 0.0, 0.0)  # Red

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, changed_positions.astype(np.float32))
        glDrawArrays(GL_POINTS, 0, count)
        glDisableClientState(GL_VERTEX_ARRAY)

        glPopAttrib()

    def draw_slice_overlay(self):
        """
        Draw the current slice image (if any) in the bottom-right corner.
        Also show simple keyboard help overlay.
        """
        if not self.slice_mode or not self.slice_surfaces or self.current_slice_index >= len(self.slice_surfaces):
            return

        surf = self.slice_surfaces[self.current_slice_index]
        size = surf.get_size()
        pixels = pygame.image.tostring(surf, "RGBA", True)

        # Save state
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        margin = 10
        x = self.width - size[0] - margin
        y = margin
        glWindowPos2f(x, y)
        glDrawPixels(size[0], size[1], GL_RGBA, GL_UNSIGNED_BYTE, pixels)

        # draw slice index text and controls hint under the slice
        lines = [
            f"Slice {self.current_slice_index+1}/{len(self.slice_surfaces)}",
            "Arrows: prev/next slice",
            "S: toggle slices, E: embed, X: extract"
        ]
        y_offset = y + size[1] + 5
        for line in lines:
            text_surface = self.stats_font.render(line, True, (255, 255, 255, 255))
            tw, th = text_surface.get_width(), text_surface.get_height()
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glWindowPos2f(x, y_offset)
            glDrawPixels(tw, th, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            y_offset += th + 2

        glPopAttrib()

    def draw_slice_plane_3d(self):
        """
        Draw the current slice outline directly on the 3D model surface
        using the actual intersection points for that slice.
        """
        if (not self.slice_mode or
            not self.model or
            self.model.slice_points_3d is None or
            self.current_slice_index >= len(self.model.slice_points_3d)):
            return

        pts3d = self.model.slice_points_3d[self.current_slice_index]
        if pts3d is None or pts3d.size == 0:
            return

        # Draw small green points along the slice intersection on the mesh
        glPushAttrib(GL_LIGHTING_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        glDisable(GL_LIGHTING)

        glPointSize(4.0)
        glColor3f(0.0, 1.0, 0.0)  # Green

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pts3d.astype(np.float32))
        glDrawArrays(GL_POINTS, 0, pts3d.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY)

        glPopAttrib()

    def get_nav_button_centers(self):
        """Calculate navigation button positions - centralized to avoid inconsistency"""
        viewport_height = self.height - self.toolbar_height
        content_area_y = 80
        slice_display_size = min(self.width - 200, viewport_height - content_area_y - 80)
        slice_y = content_area_y + ((viewport_height - content_area_y - 80) - slice_display_size) // 2
        
        button_size = 70
        button_radius = button_size // 2
        
        # Button positions (centered vertically with slice)
        cy = slice_y + slice_display_size // 2
        left_button_center = (30 + button_radius, cy)
        right_button_center = (self.width - 30 - button_radius, cy)
        
        return left_button_center, right_button_center, button_radius

    def draw_stats_on_surface(self, surface, viewport_height):
        """Draw FPS/stats counter on a pygame surface (for 2D slice view) - matches 3D layout"""
        fps = self.clock.get_fps()
        
        if self.model and self.model.triangle_count > 0:
            vertices = self.model.vertex_count
            faces = self.model.triangle_count
        else:
            vertices = 8
            faces = 12
        
        lines = [
            f"FPS: {fps:.1f}",
            f"Vertices: {vertices:,}",
            f"Faces: {faces:,}"
        ]
        
        margin = 10
        
        # Start from top and go down (same as 3D scene)
        y_offset = margin
        
        for line in lines:
            text_surface = self.stats_font.render(line, True, (255, 255, 255, 255))
            surface.blit(text_surface, (margin, y_offset))
            y_offset += text_surface.get_height() + 2

    def draw_2d_slice_view(self):
        """Draw 2D slice view - removed navigation buttons, using keyboard arrows instead"""
        if not self.slice_surfaces:
            return
        
        viewport_height = self.height - self.toolbar_height
        num_slices = len(self.slice_surfaces)
        
        view_surf = pygame.Surface((self.width, viewport_height), pygame.SRCALPHA)
        
        # Background gradient
        for y in range(viewport_height):
            factor = 0.92 + (y / viewport_height) * 0.08
            r = int(25 * factor)
            g = int(25 * factor)
            b = int(30 * factor)
            pygame.draw.line(view_surf, (r, g, b, 255), (0, y), (self.width, y))
        
        # Header
        title_font = pygame.font.Font(None, 32)
        title_text = title_font.render("Slice View", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, 20))
        view_surf.blit(title_text, title_rect)
        
        # Count changed vertices
        changed_count = 0
        total_count = 0
        if (self.model and hasattr(self.model, 'slice_points') and 
            self.current_slice_index < len(self.model.slice_points)):
            slice_vertex_indices = self.model._build_slice_vertex_indices()
            if self.current_slice_index < len(slice_vertex_indices):
                v_indices = slice_vertex_indices[self.current_slice_index]
                total_count = len(v_indices)
                if self.model.changed_vertex_mask is not None:
                    for vid in v_indices:
                        if vid < len(self.model.changed_vertex_mask) and self.model.changed_vertex_mask[vid]:
                            changed_count += 1
        
        # Info text
        info_font = pygame.font.Font(None, 24)
        idx_text = f"Slice {self.current_slice_index + 1} of {num_slices}"
        vtx_text = f"Vertices: {total_count}"
        changed_text = f"Changed: {changed_count}"
        nav_text = f"Slice axis: {self.model.slice_axis} | Use ← → arrow keys to navigate"
        
        idx_surf = info_font.render(idx_text, True, (180, 210, 255))
        vtx_surf = info_font.render(vtx_text, True, (180, 210, 255))
        changed_surf = info_font.render(changed_text, True, (255, 100, 100))
        nav_surf = info_font.render(nav_text, True, (150, 150, 150))
        
        spacing = 30
        total_width = idx_surf.get_width() + spacing + vtx_surf.get_width() + spacing + changed_surf.get_width()
        start_x = (self.width - total_width) // 2
        
        view_surf.blit(idx_surf, (start_x, 50))
        view_surf.blit(vtx_surf, (start_x + idx_surf.get_width() + spacing, 50))
        view_surf.blit(changed_surf, (start_x + idx_surf.get_width() + spacing + vtx_surf.get_width() + spacing, 50))
        
        # Navigation hint at bottom center
        nav_rect = nav_surf.get_rect(center=(self.width // 2, viewport_height - 30))
        view_surf.blit(nav_surf, nav_rect)
        
        # Slice display
        header_height = 80
        content_start = header_height
        bottom_margin = 60  # Space for navigation hint
        content_height = viewport_height - content_start - bottom_margin
        slice_size = min(self.width - 200, content_height)
        slice_x = (self.width - slice_size) // 2
        slice_y = content_start + (content_height - slice_size) // 2
        
        container_rect = pygame.Rect(slice_x - 10, slice_y - 10, slice_size + 20, slice_size + 20)
        pygame.draw.rect(view_surf, (20, 20, 25, 255), container_rect)
        pygame.draw.rect(view_surf, (80, 80, 90, 255), container_rect, 2)
        
        if 0 <= self.current_slice_index < num_slices:
            slice_surf = self.slice_surfaces[self.current_slice_index]
            if slice_surf:
                scaled_slice = pygame.transform.scale(slice_surf, (slice_size, slice_size))
                view_surf.blit(scaled_slice, (slice_x, slice_y))
                slice_border = pygame.Rect(slice_x, slice_y, slice_size, slice_size)
                pygame.draw.rect(view_surf, (100, 150, 200, 255), slice_border, 1)
        
        # Draw FPS/stats
        self.draw_stats_on_surface(view_surf, viewport_height)
        
        # Render to OpenGL
        glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        import OpenGL.GL as _gl
        _gl.glViewport(0, 0, self.width, viewport_height)
        glClear(GL_COLOR_BUFFER_BIT)
        
        pixels = pygame.image.tostring(view_surf, "RGBA", True)
        glWindowPos2f(0, 0)
        glDrawPixels(self.width, viewport_height, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
        
        glPopAttrib()

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # If in slice mode, show 2D slice view instead of 3D
        if self.slice_mode and self.slice_view_2d:
            self.draw_2d_slice_view()
        else:
            # Normal 3D view
            glLoadIdentity()
            glTranslatef(0.0, 0.0, self.zoom_z)
            glMultMatrixf(self.rotation_matrix)

            try:
                if self.model and self.model.triangle_count > 0:
                    self.model.draw()
                    # Show the 3D slice plane if slice mode is active (but not in 2D view)
                    if self.slice_mode and not self.slice_view_2d:
                        self.draw_slice_plane_3d()
                    # Draw vertices on top if enabled
                    if self.show_vertices:
                        self.draw_vertices()
                    if self.show_changed_vertices:
                        self.draw_changed_vertices()
                else:
                    draw_cube()
            except Exception as e:
                print("[Render] Exception during draw:", e)
            
            # Draw the stats overlay after the main 3D scene
            self.draw_stats_overlay()

            # Draw 2D slice preview / UI overlay if enabled (only in 3D view)
            if not self.slice_view_2d:
                self.draw_slice_overlay()
        
        # Draw toolbar at the bottom
        self.draw_toolbar()

        pygame.display.flip()

    def main_loop(self):
        while True:
            try:
                self.handle_events()
                self.draw()
                # Tick the clock to enforce the 60 FPS limit and calculate the actual FPS
                self.clock.tick(60) 
            except Exception as e:
                print("[MainLoop] Exception:", e)
                # continue

if __name__ == "__main__":
    Viewer()