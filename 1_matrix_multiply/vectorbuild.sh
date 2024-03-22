rm ./bin/main
g++ main.cpp -o ./bin/main -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -O3 -fomit-frame-pointer  -ffast-math