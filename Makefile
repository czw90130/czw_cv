#Makefile

maker = g++
link_cfg = -L/lib -L/usr/lib -L/usr/local/lib -I./include -I/usr/local/include/opencv -lm -lopencv_core -lopencv_highgui -DCV 
objs_for_main =  bin/mymat.o bin/mymat_cv.o bin/canny.o bin/sift.o  bin/liftwave.o bin/hough.o bin/main.o

allprodone : bin/test
.PHONY : allprodone
	@echo "\n---All program done... \n"

bin/test : $(objs_for_main) 
	@echo "\n*********************bins************************\n"
	$(maker) $(objs_for_main) $(link_cfg) -o bin/test
	@echo "\n---Done...\n"

bin/main.o : src/main.c bin/mymat.o include/config.h
	@echo "\n***********************main.o**************************\n"
	$(maker) -c src/main.c  $(link_cfg) -o bin/main.o
	@echo "\n---Done...\n"	

bin/mymat.o : src/mymat.c include/mymat.h
	@echo "\n******************mymat.o********************\n"
	$(maker) -c src/mymat.c  $(link_cfg) -o bin/mymat.o 
	@echo "\n---Done...\n"

bin/mymat_cv.o : src/mymat_cv.cpp include/mymat.h
	@echo "\n******************mymat_cv.o********************\n"
	$(maker) -c src/mymat_cv.cpp $(link_cfg) -o bin/mymat_cv.o 
	@echo "\n---Done...\n"

bin/canny.o : src/canny.c include/canny.h
	@echo "\n******************canny.o********************\n"
	$(maker) -c src/canny.c $(link_cfg) -o bin/canny.o 
	@echo "\n---Done...\n"

bin/sift.o : src/sift.c include/sift.h
	@echo "\n******************sift.o********************\n"
	$(maker) -c src/sift.c $(link_cfg) -o bin/sift.o 
	@echo "\n---Done...\n"

bin/liftwave.o : src/liftwave.c include/liftwave.h
	@echo "\n******************liftwave.o********************\n"
	$(maker) -c src/liftwave.c $(link_cfg) -o bin/liftwave.o 
	@echo "\n---Done...\n"

bin/hough.o : src/hough.c include/hough.h
	@echo "\n******************hough.o********************\n"
	$(maker) -c src/hough.c $(link_cfg) -o bin/hough.o 
	@echo "\n---Done...\n"


.PHONY : clean
clean :
	@echo "\n---Clean all object files...\n"
	@-rm -v  bin/*.o
	@echo "\n---Done...\n"
