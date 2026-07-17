#!/bin/bash -e

# TODO: Review and if possible fix shellcheck errors.
# shellcheck disable=all

[ "${BASH_SOURCE[0]}" ] && SCRIPT_NAME="${BASH_SOURCE[0]}" || SCRIPT_NAME=$0
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_NAME")/.." && pwd -P)"

tblite_ver="0.6.0"
tblite_sha256="372281aedb89234168d00eb691addb303197a9462a9c55d145c835f2cf5e8b42"
tblite_sdftd3_ver="1.4.0"
tblite_dftd4_ver="4.2.0"
save_tblite_rev="15915c9435644eb257178ca8f8bf7220c38b1a84"
save_tblite_repo="${SAVE_TBLITE_REPOSITORY:-https://github.com/DCM-Uni-Paderborn/save_tblite.git}"
save_tblite_src_dir="save_tblite-${save_tblite_rev}"

source "${SCRIPT_DIR}"/common_vars.sh
source "${SCRIPT_DIR}"/tool_kit.sh
source "${SCRIPT_DIR}"/signal_trap.sh
source "${INSTALLDIR}"/toolchain.conf
source "${INSTALLDIR}"/toolchain.env

[ -f "${BUILDDIR}/setup_tblite" ] && rm "${BUILDDIR}/setup_tblite"

! [ -d "${BUILDDIR}" ] && mkdir -p "${BUILDDIR}"
cd "${BUILDDIR}"
tblite_source_revision="external"

case "$with_tblite" in
  __DONTUSE__) ;;

  __INSTALL__)
    if [ "${tblite_provider}" = "save" ]; then
      tblite_source_revision="${save_tblite_rev}"
      echo "==================== Installing save_tblite ===================="
      pkg_install_dir="${INSTALLDIR}/save_tblite-${save_tblite_rev:0:12}"
      install_lock_file="${pkg_install_dir}/install_successful"
      if verify_checksums "${install_lock_file}"; then
        echo "save_tblite-${save_tblite_rev:0:12} is already installed, skipping it."
      else
        rm -rf "${save_tblite_src_dir}" "${pkg_install_dir}"
        git clone --no-checkout "${save_tblite_repo}" "${save_tblite_src_dir}"
        git -C "${save_tblite_src_dir}" checkout --detach "${save_tblite_rev}"
        cd "${save_tblite_src_dir}"

        mkdir -p build && cd build
        cmake \
          -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}" \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DCMAKE_VERBOSE_MAKEFILE=ON \
          -DBUILD_TESTING=OFF \
          -DWITH_TESTS=OFF \
          -DWITH_OpenMP=ON \
          -DWITH_DDX=OFF \
          -Dtblite-dependency-method=fetch \
          .. \
          > cmake.log 2>&1 || tail_excerpt cmake.log
        make install -j $(get_nprocs) > make.log 2>&1 || tail_excerpt make.log
        echo "${save_tblite_rev}" > save_tblite_revision
        write_checksums "${install_lock_file}" \
          "${SCRIPT_DIR}/stage8/$(basename ${SCRIPT_NAME})" \
          save_tblite_revision
        cd ../..
      fi
    else
      tblite_source_revision="${tblite_ver}"
      echo "==================== Installing tblite ===================="
      pkg_install_dir="${INSTALLDIR}/tblite-${tblite_ver}"
      install_lock_file="${pkg_install_dir}/install_successful"
      if verify_checksums "${install_lock_file}"; then
        echo "tblite-${tblite_ver} is already installed, skipping it."
      else
        retrieve_package "${tblite_sha256}" "tblite-${tblite_ver}.tar.xz"
        [ -d tblite-${tblite_ver} ] && rm -rf tblite-${tblite_ver}
        tar -xJf tblite-${tblite_ver}.tar.xz
        cd tblite-${tblite_ver}

        patch -l -d subprojects/s-dftd3 -p1 < "${SCRIPT_DIR}/stage8/simple-dftd3-${tblite_sdftd3_ver}-gradient-fixes.patch" \
          > simple_dftd3_gradient_fixes.patch.log 2>&1 || tail_excerpt simple_dftd3_gradient_fixes.patch.log
        patch -l -d subprojects/dftd4 -p1 < "${SCRIPT_DIR}/stage8/dftd4-${tblite_dftd4_ver}-gradient-fixes.patch" \
          > dftd4_gradient_fixes.patch.log 2>&1 || tail_excerpt dftd4_gradient_fixes.patch.log

        mkdir -p build && cd build
        cmake \
          -DCMAKE_INSTALL_PREFIX="${pkg_install_dir}" \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DCMAKE_VERBOSE_MAKEFILE=ON \
          -DBUILD_TESTING=OFF \
          -DWITH_TESTS=OFF \
          .. \
          > cmake.log 2>&1 || tail_excerpt cmake.log
        make install -j $(get_nprocs) > make.log 2>&1 || tail_excerpt make.log
        write_checksums "${install_lock_file}" "${SCRIPT_DIR}/stage8/$(basename ${SCRIPT_NAME})" \
          "${SCRIPT_DIR}/stage8/simple-dftd3-${tblite_sdftd3_ver}-gradient-fixes.patch" \
          "${SCRIPT_DIR}/stage8/dftd4-${tblite_dftd4_ver}-gradient-fixes.patch"
        cd ..
      fi
    fi
    ;;
  __SYSTEM__)
    echo "==================== Finding tblite from system paths ===================="
    check_pkgconfig tblite
    pkg_install_dir="$(pkg-config --variable=prefix tblite)"
    ;;
  *)
    echo "==================== Linking TBLITE to user paths ===================="
    pkg_install_dir="${with_tblite}"
    check_dir "${pkg_install_dir}/include"
    ;;
esac

if [ "$with_tblite" != "__DONTUSE__" ]; then
  cat << EOF > "${BUILDDIR}/setup_tblite"
export TBLITE_ROOT="${pkg_install_dir}"
export TBLITE_VER="${tblite_ver}"
export TBLITE_PROVIDER="${tblite_provider}"
export SAVE_TBLITE_REVISION="${save_tblite_rev}"
export TBLITE_SOURCE_REVISION="${tblite_source_revision}"
EOF
  if [ "$with_tblite" != "__SYSTEM__" ]; then
    cat << EOF >> "${BUILDDIR}/setup_tblite"
prepend_path PATH "${pkg_install_dir}/bin"
prepend_path LD_LIBRARY_PATH "${pkg_install_dir}/lib"
prepend_path LD_RUN_PATH "${pkg_install_dir}/lib"
prepend_path LIBRARY_PATH "${pkg_install_dir}/lib"
prepend_path PKG_CONFIG_PATH "${pkg_install_dir}/lib/pkgconfig"
prepend_path CMAKE_PREFIX_PATH "${pkg_install_dir}"
EOF
  fi
  filter_setup "${BUILDDIR}/setup_tblite" "${SETUPFILE}"
fi

load "${BUILDDIR}/setup_tblite"
write_toolchain_env "${INSTALLDIR}"

cd "${ROOTDIR}"
report_timing "tblite"
