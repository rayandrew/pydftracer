import os


def get_dftracer_preload_path():
    """Find the path to libdftracer_preload.so in the virtual environment"""
    try:
        import site

        # Get all site-packages directories
        site_packages_dirs = site.getsitepackages()

        # Also check the user site-packages if it exists
        try:
            site_packages_dirs.append(site.getusersitepackages())
        except AttributeError:
            pass

        # Check for dftracer_libs in each site-packages directory
        for site_dir in site_packages_dirs:
            dftracer_libs_path = os.path.join(site_dir, "dftracer_libs")

            if os.path.isdir(dftracer_libs_path):
                # Try both possible locations
                lib64_path = os.path.join(
                    dftracer_libs_path, "lib64", "libdftracer_preload.so"
                )
                lib_path = os.path.join(
                    dftracer_libs_path, "lib", "libdftracer_preload.so"
                )

                if os.path.exists(lib64_path):
                    return lib64_path
                elif os.path.exists(lib_path):
                    return lib_path

        print("Warning: dftracer_libs directory or libdftracer_preload.so not found")
        return ""

    except Exception as e:
        print(f"Warning: Error finding dftracer_libs: {e}")
        return ""
