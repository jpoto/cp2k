## run toolchain script
cd tools/toolchain
./install_cp2k_toolchain.sh -j 16 \
    --enable-cuda=yes \
    --gpu-ver=A100 \
    --with-tblite=install \
    --with-gcc=system \
    --with-cmake=system \
    --with-ninja=system \
    --with-openblas=install \
    --with-libxc=install \
    --with-libint=install \
    --with-fftw=install \
    --with-libxsmm=install \
    --with-spglib=install \
    --with-elpa=no
