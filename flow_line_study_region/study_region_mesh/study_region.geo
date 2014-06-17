lc = 500;

Point(1) = {423863.131,-1304473.006, 0, lc}; // lower left 
Point(2) = {442739.172,-1304473.006, 0, lc}; // lower right
Point(3) = {442739.172,-1285920.863, 0, lc}; // upper right
Point(4) = {423863.131,-1285920.863, 0, lc}; // upper left

Line(1) = {1,2} ;
Line(2) = {2,3} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;

Line Loop(5) = {4,1,2,3} ;

Plane Surface(6) = {5} ;

//h = 1;
//Extrude {0,0,h} {
//  Surface{6}; Layers{10};}
