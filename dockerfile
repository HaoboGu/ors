FROM rustembedded/cross:x86_64-unknown-linux-gnu-0.2.1

RUN dpkg --add-architecture arm && \
    apt-get update && \
    apt-get install -y apt-transport-https wget && \
    (wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -) && \
    printf "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-12 main\ndeb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-12 main" > /etc/apt/sources.list.d/backports.list && \
    apt-get update && \
    apt-get install -y llvm-3.9-dev libclang-3.9-dev clang-3.9 && \
    apt-get purge -y apt-transport-https wget