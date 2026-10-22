"""Conservative neutral-white RGB contour parameters for the current plain white ball asset.

Inputs are an RGB uint8 image and YOLO xyxy box in that ORIGINAL image.
No camera/ground/world geometry, object truth, temporal borrowing, GPU, or
circle completion. A returned curve consists only of observed mask-boundary
pixel centres. Shape fitting is a rejection check, never contour replacement.
This is not validated for patterned real balls, arbitrary lighting or hardware.
"""
import math
import cv2
import numpy as np

VERSION = 'neutral_white_ball_RGB_contour_parameters_v2'
PARAMETERS = {
    'roi_pad_fraction': .15, 'roi_pad_min_px': 4, 'image_margin_px': 2,
    'white_value_min': 90, 'white_chroma_over_value_max': .035,
    'green_G_over_R_min': 1.12, 'green_G_over_B_min': 1.07, 'green_delta_min': 10,
    'contour_min_points': 24, 'component_min_area_px': 30,
    'area_over_detection_box_min': .35, 'area_over_detection_box_max': 1.20,
    'secondary_component_area_fraction': .25, 'hole_area_fraction_max': .02,
    'solidity_min': .96, 'circularity_min': .78, 'ellipse_axis_ratio_min': .72,
    'ellipse_radial_P95_max': .035, 'ellipse_radial_max': .10,
    'boundary_background_ring_px': 4, 'nonwhite_ring_fraction_min': .70,
    'ring_angular_bins': 24, 'nonwhite_bin_fraction_min': .35,
    'min_nonwhite_bins': 22,
}


def extract_white_ball_contour(rgb_uint8_HWC, bbox_xyxy_px):
    """Return accepted/reason/closed original-pixel contour/diagnostics.

    Rejected parameterss always return an empty contour. There is no repair,
    inpainting, hull replacement, ellipse replacement or missing-arc synthesis.
    Pixels may differ from the true optical silhouette by sampling/segmentation;
    downstream ray geometry must retain that uncertainty.
    """
    result = {'kind': VERSION, 'accepted': False, 'reason': None,
              'contour_xy_px': [], 'bbox_xyxy_px': None, 'roi_xyxy_px': None,
              'diagnostics': {}, 'parameters': PARAMETERS.copy(),
              'contour_is_observed_pixel_centres': True,
              'missing_boundary_reconstructed': False, 'geometry_or_truth_used': False}
    def reject(reason):
        result['reason'] = reason
        return result
    rgb = np.asarray(rgb_uint8_HWC)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return reject('requires_original_RGB_uint8_HWC')
    try:
        box = np.asarray(bbox_xyxy_px, dtype=np.float64)
    except (TypeError, ValueError):
        return reject('invalid_bbox')
    if box.shape != (4,) or not np.isfinite(box).all(): return reject('invalid_bbox')
    x1, y1, x2, y2 = box.tolist(); result['bbox_xyxy_px'] = box.tolist()
    h, w = rgb.shape[:2]
    if x2 <= x1 or y2 <= y1: return reject('invalid_bbox_extent')
    margin = PARAMETERS['image_margin_px']
    if x1 <= margin or y1 <= margin or x2 >= w-1-margin or y2 >= h-1-margin:
        return reject('detection_touches_image_edge')
    pad = max(PARAMETERS['roi_pad_min_px'], PARAMETERS['roi_pad_fraction']*max(x2-x1, y2-y1))
    left, top = max(0, math.floor(x1-pad)), max(0, math.floor(y1-pad))
    right, bottom = min(w, math.ceil(x2+pad)), min(h, math.ceil(y2+pad))
    result['roi_xyxy_px'] = [left, top, right, bottom]
    crop = rgb[top:bottom, left:right].astype(np.float32)
    red, green, blue = crop[:,:,0], crop[:,:,1], crop[:,:,2]
    value, low = crop.max(axis=2), crop.min(axis=2)
    field = ((green > red*PARAMETERS['green_G_over_R_min']) &
             (green > blue*PARAMETERS['green_G_over_B_min']) &
             (green-np.minimum(red,blue) >= PARAMETERS['green_delta_min']))
    white = ((value >= PARAMETERS['white_value_min']) &
             (value-low <= PARAMETERS['white_chroma_over_value_max']*value) & ~field)
    # No dilation/closing of white pixels: never fill an occluded ball boundary.
    contours, hierarchy = cv2.findContours(white.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None: return reject('no_neutral_white_component')
    parameterss = []
    for index, contour in enumerate(contours):
        if hierarchy[0,index,3] != -1: continue
        area = cv2.contourArea(contour)
        moment = cv2.moments(contour)
        if area < PARAMETERS['component_min_area_px'] or moment['m00'] == 0: continue
        cx, cy = moment['m10']/moment['m00']+left, moment['m01']/moment['m00']+top
        if x1 <= cx <= x2 and y1 <= cy <= y2: parameterss.append((area,index,contour,cx,cy))
    if not parameterss: return reject('no_component_centred_in_detection')
    parameterss.sort(key=lambda x:(-x[0],x[1]))
    area,index,contour,cx,cy = parameterss[0]
    diag = result['diagnostics']
    diag.update(parameters_count=len(parameterss), area_px=area, observed_component_centroid_xy_px=[cx,cy])
    if len(parameterss)>1 and parameterss[1][0] >= area*PARAMETERS['secondary_component_area_fraction']:
        return reject('multiple_plausible_white_components')
    points = contour[:,0,:].astype(np.float64)
    if len(points) < PARAMETERS['contour_min_points']: return reject('contour_too_small')
    if np.any(points[:,0] <= 0) or np.any(points[:,1] <= 0) or np.any(points[:,0] >= right-left-1) or np.any(points[:,1] >= bottom-top-1):
        return reject('component_touches_ROI_edge')
    world_points = points+np.asarray([left,top])
    if np.any(world_points[:,0] <= margin) or np.any(world_points[:,1] <= margin) or np.any(world_points[:,0] >= w-1-margin) or np.any(world_points[:,1] >= h-1-margin):
        return reject('component_touches_image_edge')
    area_ratio = area/((x2-x1)*(y2-y1)); diag['area_over_bbox'] = area_ratio
    if not PARAMETERS['area_over_detection_box_min'] <= area_ratio <= PARAMETERS['area_over_detection_box_max']:
        return reject('component_detection_area_mismatch')
    holes = sum(cv2.contourArea(c) for k,c in enumerate(contours) if hierarchy[0,k,3] == index)
    diag['hole_area_fraction'] = holes/area
    if holes/area > PARAMETERS['hole_area_fraction_max']: return reject('large_dark_hole_or_occluder')
    perimeter = cv2.arcLength(contour,True)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    circularity = 4*math.pi*area/(perimeter*perimeter)
    solidity = area/hull_area if hull_area else 0.
    diag.update(circularity=circularity, solidity=solidity)
    if solidity < PARAMETERS['solidity_min'] or circularity < PARAMETERS['circularity_min']:
        return reject('nonconvex_or_unreliable_round_shape')
    (ex,ey),(a,b),angle = cv2.fitEllipse(contour)
    ratio = min(a,b)/max(a,b)
    theta=math.radians(angle); c,s=math.cos(theta),math.sin(theta)
    xy=points-np.asarray([ex,ey]);u=c*xy[:,0]+s*xy[:,1];v=-s*xy[:,0]+c*xy[:,1]
    radial=np.abs(np.sqrt((u/(a/2))**2+(v/(b/2))**2)-1)
    diag.update(ellipse_axis_ratio=ratio, ellipse_radial_P95=float(np.quantile(radial,.95)),
                ellipse_radial_max=float(radial.max()), ellipse_is_only_rejection_check=True)
    if ratio < PARAMETERS['ellipse_axis_ratio_min'] or np.quantile(radial,.95)>PARAMETERS['ellipse_radial_P95_max'] or radial.max()>PARAMETERS['ellipse_radial_max']:
        return reject('partial_occlusion_or_nonelliptic_boundary')
    # Exterior must differ from the neutral-white foreground; the current
    # cyan/grey pitch and tinted white lines are not necessarily green in RGB.
    # This is a contrast check, not an obstacle-free or semantic claim.
    # Dilate only for inspecting the exterior background; returned contour stays
    # the unmodified observed mask boundary. The filled diagnostic region is not
    # passed downstream and does not synthesize a spherical silhouette.
    filled=np.zeros(white.shape,dtype=np.uint8);cv2.drawContours(filled,[contour],-1,1,-1)
    rad=PARAMETERS['boundary_background_ring_px']
    ring=(cv2.dilate(filled,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(rad*2+1,rad*2+1)))>0)&(filled==0)
    yy,xx=np.nonzero(ring); contrast_values=(~white)[ring]
    if len(xx)==0: return reject('missing_exterior_background')
    fractions=[]; bins=PARAMETERS['ring_angular_bins']
    angles=(np.arctan2(yy-ey,xx-ex)+2*np.pi)%(2*np.pi)
    sector=np.minimum((angles/(2*np.pi)*bins).astype(int),bins-1)
    for k in range(bins):
        selected=contrast_values[sector==k]
        fractions.append(float(selected.mean()) if len(selected) else 0.)
    contrast_count=sum(x>=PARAMETERS['nonwhite_bin_fraction_min'] for x in fractions)
    diag.update(exterior_nonwhite_fraction=float(contrast_values.mean()), exterior_nonwhite_bins=contrast_count,
                exterior_nonwhite_bin_fractions=fractions)
    if contrast_values.mean()<PARAMETERS['nonwhite_ring_fraction_min'] or contrast_count<PARAMETERS['min_nonwhite_bins']:
        return reject('unsupported_background_or_possible_occlusion')
    # Explicitly close by repeating the first EXISTING boundary pixel.
    closed=np.vstack([world_points,world_points[0]])
    result.update(accepted=True, reason='observed_closed_white_ball_parameters', contour_xy_px=closed.tolist())
    diag['boundary_points_excluding_repeated_endpoint']=len(points)
    diag['pixel_boundary_quantization_note']='Observed foreground pixel centres, not subpixel tangent rays; retain pixel uncertainty downstream.'
    return result
