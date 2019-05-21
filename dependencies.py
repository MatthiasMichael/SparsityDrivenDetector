# The way this project is set up, managing dependencies can be hard
# This file should bring a bit more clarity into the system by 
# making all dependencies visible and ordering them by their level

def get(*args):
	deps = []
	for arg in args:
		deps = deps + arg
	
	return tuple(set(deps))

conflict_resolve = ["OpenSSL/1.0.2r@conan/stable"]

environment = [
	"Geometry/1.0@MatthiasMichael/stable", 
	"Roi3DF/1.0@MatthiasMichael/stable",
	"boost_filesystem/1.66.0@bincrafters/stable",
	"libzip/1.5.1@bincrafters/stable"
]

sdd = [
	"CPLEX/12.7.1@MatthiasMichael/stable", 
	"opencv/3.4.5@conan/stable"
]

unreal_input = [
	"opencv/3.4.5@conan/stable",
	"ffmpeg/4.0.2@bincrafters/stable",
	"Geometry/1.0@MatthiasMichael/stable", 
	"sqlite3/3.21.0@bincrafters/stable"
]

visualization = [
	"OsgVisualization/1.0@MatthiasMichael/stable",
	"Geometry/1.0@MatthiasMichael/stable"
]

shape_from_silhouette_common = [
	"Roi3DF/1.0@MatthiasMichael/stable",
]

shape_from_silhouette = [
	"opencv/3.4.5@conan/stable"
]

controller = [
	"boost_program_options/1.66.0@bincrafters/stable"
]

all = get(conflict_resolve, environment, sdd, unreal_input, visualization, shape_from_silhouette_common, shape_from_silhouette, controller)

