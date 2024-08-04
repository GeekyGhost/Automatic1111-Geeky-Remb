import launch

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "requirement for Geeky RemB")

if not launch.is_installed("opencv-python"):
    launch.run_pip("install opencv-python", "requirement for Geeky RemB")