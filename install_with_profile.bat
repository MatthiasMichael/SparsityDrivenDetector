conan install . -if %1 -pr %2 --build missing

set projects=^
timer ^
environment ^
sparsity_driven_detector ^
shape_from_silhouette_common ^
shape_from_silhouette ^
shape_from_silhouette_cuda ^
shape_from_silhouette_bridge ^
unreal_input ^
utility ^
fusion ^
visualization_sparsity_driven_detector ^
visualization_environment ^
visualization_shape_from_silhouette ^
visualization_fusion ^
interface ^
controller ^
experiment ^
executable

(for %%a in (%projects%) do (
	IF NOT EXIST %1.%%a (
		mkdir %1.%%a
	)
	conan install %%a -if %1.%%a -pr %2 --build missing
))

mkdir %1
conan build . -bf %1 -c