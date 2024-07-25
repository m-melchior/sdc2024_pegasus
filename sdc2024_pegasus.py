#!/usr/bin/env python

import carb
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.kit
import omni.timeline

from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension
from omni.isaac.core.utils.prims import move_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading, update_stage_async, update_stage
from omni.isaac.core.materials import OmniPBR, PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim
from omni.isaac.core.utils import prims
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.objects import FixedCuboid
from omni.isaac.core.physics_context import PhysicsContext
from omni.kit.material.library import CreateAndBindMdlMaterialFromLibrary
from omni.isaac.version import get_version

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs import ROS2Camera
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import json
import math
import numpy
import os.path
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdShade, UsdGeom
from scipy.spatial.transform import Rotation
from typing import Sequence, Tuple

# # 18.15
WIDTH_ARENA = 3 * 6.05
# # 33,25
LENGTH_ARENA = 5 * 6.65

# HEIGHT_WALL = 8
# SIZE_PILLAR = 0.5

POS_START_LINE = 1.5

NUM_VEHICLES = 1
DIST_Y = 2

INDEX_MARKER_ID 			= 0
INDEX_MARKER_TRANSLATION 	= 1
INDEX_MARKER_ROTATION 		= 2

INDEX_MATERIAL_NAME 		= 0
INDEX_MATERIAL_PATH			= 1

INDEX_OBJECT_TYPE 			= 0
INDEX_OBJECT_NAME			= 1
INDEX_OBJECT_TRANSLATION	= 2
INDEX_OBJECT_SCALE			= 3
INDEX_OBJECT_COLOR			= 4

PATH_PARAMETERS = "./assets/parameters.json"

PATH_MARKER_SIGNS = "./assets/marker_signs.json"
PATH_MARKER_BOXES = "./assets/marker_boxes.json"
marker_list = [
	["sign", PATH_MARKER_SIGNS],
	["box", PATH_MARKER_BOXES]
]

PATH_SCENERY_BOXES = "./assets/scenery_boxes.json"
PATH_SCENERY = "./assets/scenery.json"

def get_data(path_in):
	data = None

	try:
		with open(path_in, "r") as file:
			data = json.load(file)
	except Exception as ex:
		print (f"ERROR: railed to get data: {e}")

	return data

def get_marker_data(path_in):
	with open(path_in, "r") as file:
		marker_data = json.load(file)

	size_markers = marker_data["size_markers"]
	path_mat = marker_data["path_mat"]
	size_tags = marker_data["size_tags"]
	markers = marker_data["markers"]

	return size_markers, path_mat, size_tags, markers

def evaluate_parametric_value(value, parameters):
    if isinstance(value, str):
        return eval(value, {}, parameters)
    return value

# ******************************************
disable_extension("omni.isaac.ros_bridge")
enable_extension("omni.isaac.ros2_bridge")


# ******************************************
def addDefaultOps(prim: Usd.Prim,
				  ) -> UsdGeom.Xform:
	xform = UsdGeom.Xform(prim)
	xform.ClearXformOpOrder()

	try:
		xform.AddTranslateOp(precision = UsdGeom.XformOp.PrecisionDouble)
	except:
		pass

	try:
		xform.AddRotateXYZOp(precision = UsdGeom.XformOp.PrecisionDouble)
	except:
		pass

	try:
		xform.AddScaleOp(precision = UsdGeom.XformOp.PrecisionDouble)
	except:
		pass

	return xform

def setDefaultOps(xform: UsdGeom.Xform,
				  translation: 	Tuple[float, float, float],
				  rotation: 	Tuple[float, float, float],
				  scale: 		Tuple[float, float, float],
				  ) -> None:
	xform = UsdGeom.Xform(xform)
	xform_ops = xform.GetOrderedXformOps()

	try:
		xform_ops[0].Set(Gf.Vec3d(float(translation[0]), float(translation[1]), float(translation[2])))
	except Exception as e:
		print(f"Failed to add Translation: {e}\n")
		pass

	try:
		xform_ops[1].Set(Gf.Vec3d(float(rotation[0]), float(rotation[1]), float(rotation[2])))
	except Exception as e:
		print(f"Failed to add Rotation: {e}\n")
		pass

	try:
		xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
	except Exception as e:
		print(f"Failed to add Scale: {e}\n")
		pass


class PegasusApp:
	# ******************************************
	def __init__(self):
		self.timeline = omni.timeline.get_timeline_interface()

		self.pg = PegasusInterface()

		self.parameters = None
		self.vehicles = []

		self.init_scenery()

		self.init_vehicles()

		self.world.reset()

		self.stop_sim = False


	# ******************************************
	def init_scenery(self):
		# self.pg._world_settings["physics_dt"] = 1.0 / 60.0
		# self.pg._world_settings["rendering_dt"] = 1.0 / 30.0
		self.pg._world = World(**self.pg._world_settings)
		self.world = self.pg.world

		attributes = {"intensity": 3000.0}
		attributes = {f"inputs:{k}": v for k, v in attributes.items()}
		prim_utils.create_prim(
			f"/World/Scenery/AmbientLight",
			"DistantLight",
			attributes = attributes,
			translation = [LENGTH_ARENA / 2, 0, 0]
		)

		PhysicsContext()

		plane = GroundPlane(
			prim_path = "/World/Scenery/ground",
			size = 100,
			color = numpy.array([0.5, 0.5, 0.5])
		)

		self.init_objects()

		for marker_type, marker_path in marker_list:
			self.init_markers(marker_type, marker_path)

	# ******************************************
	def init_objects(self):
		working_path = os.getcwd()

		parameters = get_data(PATH_PARAMETERS)
		scenery_data = get_data(PATH_SCENERY)

		materials = {}
		for material_data in scenery_data["materials"]:
			material_name, material_path = material_data

			material_path = working_path + material_path

			materials[material_name] = OmniPBR(
											material_name,
											texture_path = material_path
										)


		index_object = 0
		for object_data in scenery_data["objects"]:
			object_type, obj_name, translation, scale, material = object_data
			if (object_type == "FixedCuboid"):
				translation = [evaluate_parametric_value(val, parameters) for val in translation]
				scale = [evaluate_parametric_value(val, parameters) for val in scale]

				print(f"{index_object}: {translation}, {scale}")


				if(isinstance(material, str)):
					FixedCuboid(
						obj_name,
						translation = translation,
						scale = scale,
						visual_material = materials[material]
					)
				else:
					color = numpy.array(material)
					FixedCuboid(
						obj_name,
						translation = translation,
						scale = scale,
						color = color
					)
			else:
				print(f"ERROR: Unknown object_type: {object_type}")

			index_object += 1

	# ******************************************
	def init_markers(self, type, path_markerdata):
		parameters = get_data(PATH_PARAMETERS)
		marker_data = get_data(path_markerdata)

		size_markers = marker_data["size_markers"]
		path_mat = marker_data["path_mat"]
		size_tags = marker_data["size_tags"]
		markers = marker_data["markers"]

		stage = omni.usd.get_context().get_stage()
		scale = [size_markers, 0.01, size_markers]

		for marker in markers:
			id_marker, translation, rotation = marker
			translation = [evaluate_parametric_value(val, parameters) for val in translation]
			print(f"marker: {id_marker}, {translation}, {rotation}")

			path_prim = f"/World/Scenery/{type}_{id_marker:02}"
			path_mdl = "./assets/material/mosaic_texture.mdl"
			name_mtl = "MosaicTexture"
			path_mtl = f"/Looks/{type}_{id_marker:02}"

			result, path_tmp = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
			omni.kit.commands.execute("MovePrim", path_from = path_tmp, path_to = path_prim)
			prim = stage.GetPrimAtPath(path_prim)
			addDefaultOps(prim)
			setDefaultOps(
				prim,
				translation = translation,
				rotation = rotation,
				scale = scale
			)

			self.material_load_and_bind(
				stage,
				path_prim,
				path_mdl,
				name_mtl,
				path_mtl,
				path_mat,
				size_tags,
				id_marker
			)


	# ******************************************
	def init_vehicles(self):

		offset = (WIDTH_ARENA - (NUM_VEHICLES -1) * DIST_Y) / 2
		# print(f"offset: {offset}")

		for index_vehicle in range(NUM_VEHICLES):
			# position = [POS_START_LINE, offset + index_vehicle * DIST_Y, 0.07]
			# rotation = Rotation.from_euler("XYZ", [0.0, 0.0, 40.0], degrees = True).as_quat()
			position = [LENGTH_ARENA / 2, 2.0, 0.07]
			print(f"position: {position}")
			rotation = Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees = True).as_quat()

			config_multirotor = MultirotorConfig()
			
			mavlink_config = MavlinkBackendConfig({
				"vehicle_id": index_vehicle,
				"px4_autolaunch": True,
				"px4_dir": self.pg.px4_path,
				"px4_vehicle_model": self.pg.px4_default_airframe
			})
			config_multirotor.backends = [MavlinkBackend(mavlink_config)]

			# config_multirotor.backends = [ROS2Backend(vehicle_id = index_vehicle, config = {"namespace": f"robot_{index_vehicle:02}"})]

						# "resolution": [1920, 1080]
						# "resolution": [800, 600]
			config_multirotor.graphs = [
				ROS2Camera(
					"body/Camera",
					config={
						"types": ['rgb', 'camera_info'],
						"resolution": [1920, 1080]
					}
				)
			]

			vehicle = Multirotor(
						f"/World/Vehicles/quadrotor_{index_vehicle:02}",
						ROBOTS["Iris"],
						index_vehicle,
						position,
						rotation,
						config = config_multirotor
			)
			self.vehicles.append(vehicle)


	def material_load_and_bind(self,
		stage_in,
		path_prim_in,
		path_mdl_in,
		name_mtl_in,
		path_mtl_in,
		path_mosaic_in,
		tag_size_in,
		id_marker_in
	):
		working_path = os.getcwd()
		# print(f"working_path: {working_path}")

		omni.kit.commands.execute(
			"CreateMdlMaterialPrimCommand",
			mtl_url = path_mdl_in,
			mtl_name = name_mtl_in,
			mtl_path = Sdf.Path(path_mtl_in),
		)

		mtl_prim = stage_in.GetPrimAtPath(path_mtl_in)
		omni.usd.create_material_input(
			mtl_prim,
			"tag_mosaic",
			working_path + path_mosaic_in,
			Sdf.ValueTypeNames.Asset
		)

		omni.usd.create_material_input(
			mtl_prim,
			"tag_size",
			tag_size_in,
			Sdf.ValueTypeNames.Int,
		)

		omni.usd.create_material_input(
			mtl_prim,
			"tags_per_row",
			10,
			Sdf.ValueTypeNames.Int,
		)

		omni.usd.create_material_input(
			mtl_prim,
			"tag_spacing",
			1,
			Sdf.ValueTypeNames.Int,
		)
		
		omni.usd.create_material_input(
			mtl_prim,
			"tag_id",
			id_marker_in,
			Sdf.ValueTypeNames.Int,
		)

		prim = stage_in.GetPrimAtPath(path_prim_in)
		shade = UsdShade.Material(mtl_prim)
		UsdShade.MaterialBindingAPI(prim).Bind(
			shade,
			UsdShade.Tokens.strongerThanDescendants
		)


	# ******************************************
	def run(self):
		self.timeline.play()

		while simulation_app.is_running() and not self.stop_sim:
			# draw.draw_lines_spline(point_list_1, (31/255, 119/255, 0/255, 1), 5, False)
			self.world.step(render = True)

		carb.log_warn("PegasusApp Simulation App is closing.")
		self.timeline.stop()
		simulation_app.close()

# ******************************************
def main():
	pg_app = PegasusApp()
	pg_app.run()

if __name__ == "__main__":
	main()