#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

xtb_ver="6.7.0"
xtb_sha256="0e3e8d5f9e9e5414b9979967c074c953706053832e551d922c27599e7324bace"

source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_xtb" ] && rm "${BUILDDIR}/setup_xtb"

XTB_DFLAGS=''
XTB_CFLAGS=''
XTB_LDFLAGS=''
XTB_LIBS=''
! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"

case "$with_xtb" in
  __DONTUSE__) ;;

  __INSTALL__)
    echo "==================== Installing GRIMME XTB ===================="
    require_env OPENBLAS_ROOT
    require_env MATH_LIBS

    pkg_install_dir="${INSTALLDIR}/xtb-${xtb_ver}"
    install_lock_file="${pkg_install_dir}/install_successful"

    if verify_checksums "${install_lock_file}"; then
      echo "xtb-${xtb_ver} is already installed, skipping it."
    else
      if [ -f xtb-${xtb_ver}.tar.gz ]; then
        echo " xtb-${xtb_ver}.tar.gz is found"
      else
        wget --quiet -O xtb-${xtb_ver}.tar.gz https://github.com/grimme-lab/xtb/archive/refs/tags/v${xtb_ver}.tar.gz
      fi

      echo "Installing from scratch into ${pkg_install_dir}"
      [ -d xtb-${xtb_ver} ] && rm -rf xtb-${xtb_ver}
      tar -xzf xtb-${xtb_ver}.tar.gz
      cd xtb-${xtb_ver}

      rm -Rf build
      mkdir build
      cd build

      CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${OPENBLAS_ROOT}" cmake -B . \
        -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}" \
        -DCMAKE_CXX_COMPILER="${MPICXX}" \
        -DCMAKE_C_COMPILER="${MPICC}" \
        -DCMAKE_Fortran_COMPILER="${MPIFC}" \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        .. \
        > cmake.log 2>&1 || tail -n ${LOG_LINES} cmake.log
      cmake --build . -j $(get_nprocs) >> build.log 2>&1 || tail -n ${LOG_LINES} build.log
      cmake --install . >> install.log 2>&1 || tail -n ${LOG_LINES} install.log

      cd ..
      echo "==================== Linking Grimme XTB to user paths ===================="
    fi
    write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage8/$(basename ${SCRIPT_NAME})"
    ;;

  __SYSTEM__)
    echo "==================== Finding xtb from system paths ===================="
    check_command pkg-config --modversion xtb
    add_include_from_paths XTB_CFLAGS "xtb.h" $INCLUDE_PATHS
    add_include_from_paths XTB_CFLAGS "mctc_io.mod" $INCLUDE_PATHS
    add_lib_from_paths XTB_LDFLAGS "libxtb.*" $LIB_PATHS
    ;;

  *)
    echo "==================== Linking xtb to user paths ===================="
    pkg_install_dir="$with_xtb"
    check_dir "${pkg_install_dir}/include"
    ;;

esac

if [ "$with_xtb" != "__DONTUSE__" ]; then

  XTB_DFLAGS="-D__XTB"
  XTB_LIBS="-lxtb -lmctc-lib"

  if [ "$with_xtb" != "__SYSTEM__" ]; then
    XTB_LOC=$(find ${pkg_install_dir}/include -name "tblite_xtb.mod")
    XTB_XTB=${XTB_LOC%/*}
    # use the lib64 directory if present
    XTB_LIBDIR="${pkg_install_dir}/lib"
    [ -d "${pkg_install_dir}/lib64" ] && XTB_LIBDIR="${pkg_install_dir}/lib64"

    XTB_CFLAGS="-I'${pkg_install_dir}/include' -I'${XTB_XTB}' -I'${XTB_MCTC}'"
    XTB_LDFLAGS="-L'${XTB_LIBDIR}' -Wl,-rpath,'${XTB_LIBDIR}'"

    cat << EOF > "${BUILDDIR}/setup_xtb"
prepend_path LD_LIBRARY_PATH "${XTB_LIBDIR}"
prepend_path LD_RUN_PATH "${XTB_LIBDIR}"
prepend_path LIBRARY_PATH "${XTB_LIBDIR}"
prepend_path CPATH "$pkg_install_dir/include"
prepend_path PKG_CONFIG_PATH "${XTB_LIBDIR}/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "$pkg_install_dir"
EOF
  fi

  cat << EOF >> "${BUILDDIR}/setup_xtb"
export XTB_XTB="${XTB_XTB}"
export XTB_MCTC="${XTB_MCTC}"
export XTB_LIBDIR="${XTB_LIBDIR}"
export XTB_INCLUDE_DIR="$pkg_install_dir/include"
export XTB_ROOT="${pkg_install_dir}"
export XTB_DFLAGS="${XTB_DFLAGS}" 
export XTB_CFLAGS="${XTB_CFLAGS}"
export XTB_LDFLAGS="${XTB_LDFLAGS}"
export XTB_LIBS="${XTB_LIBS}"
export CP_DFLAGS="\${CP_DFLAGS} \${XTB_XTB}"
export CP_CFLAGS="\${CP_CFLAGS} \${XTB_CFLAGS}"
export CP_LDFLAGS="\${CP_LDFLAGS} \${XTB_LDFLAGS}"
export CP_LIBS="\${XTB_LIBS} \${CP_LIBS}"
EOF
  cat "${BUILDDIR}/setup_xtb" >> $SETUPFILE
fi

load "${BUILDDIR}/setup_xtb"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "xtb"
