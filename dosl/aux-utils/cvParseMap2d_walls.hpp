/** **************************************************************************************
*                                                                                        *
*    Part of                                                                             *
*    Discrete Optimal search Library (DOSL)                                              *
*    A template-based C++ library for discrete search                                    *
*    Version 3.x                                                                         *
*    ----------------------------------------------------------                          *
*    Copyright (C) 2017  Subhrajit Bhattacharya                                          *
*                                                                                        *
*    This program is free software: you can redistribute it and/or modify                *
*    it under the terms of the GNU General Public License as published by                *
*    the Free Software Foundation, either version 3 of the License, or                   *
*    (at your option) any later version.                                                 *
*                                                                                        *
*    This program is distributed in the hope that it will be useful,                     *
*    but WITHOUT ANY WARRANTY; without even the implied warranty of                      *
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                       *
*    GNU General Public License for more details <http://www.gnu.org/licenses/>.         *
*                                                                                        *
*                                                                                        *
*    Contact:  subhrajit@gmail.com                                                       *
*              https://www.lehigh.edu/~sub216/ , http://subhrajit.net/                   *
*                                                                                        *
*                                                                                        *
*************************************************************************************** **/

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

// Other libraries:
// Open CV:
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui.hpp> 

// DOSL library

#ifndef _DOSL_ALGORITHM // can pass at command line during compilation: -D_DOSL_ALGORITHM=AStar
    #define _DOSL_ALGORITHM  AStar
#endif

#include <dosl/dosl>

// Local libraries/headers
#include <dosl/aux-utils/cvParseMap2d.hpp>
#include <dosl/aux-utils/double_utils.hpp>
#include <dosl/aux-utils/string_utils.hpp> // compute_program_path
#include "../../include-local/RSJparser.tcc"

// =======================
// Other parameters

#define GRAPH_TYPE 6 // 6 or 8
#define PROB_HMTPY 1

/* // math macros / constants
#define sign(x) ((x>0.0)?1.0:((x<0.0)?-1.0:0.0))
#define PI       3.14159265359
#define PI_BY_3  1.0471975512
#define SQRT3BY2 0.86602540378
#define INFINITESIMAL_DOUBLE  1e-6 */

// output options
#define _STAT 0

#define _VIS 1
#define VIS_INTERVAL 100
#define VERTEX_COLORS 1
#define SAVE_IMG_INTERVAL 10000 // 0 to not save at all. -1 to save last frame only.

// ---------------------------------------------------

#define COORD_TYPE double

double wiggle = 1e-3;

// ==============================================================================

// A node of the graph
class myNode : public _DOSL_ALGORITHM::Node<myNode,double>
{
public:
    COORD_TYPE x, y;
    std::vector<int> h;
    
    #if GRAPH_TYPE == 8
        void put_in_grid (void) { }
    #elif GRAPH_TYPE == 6
        void put_in_grid (void) {
            int yLevel = round (y / SQRT3BY2);
            y = SQRT3BY2*yLevel;
            if (yLevel%2 == 0) // even
                x = round(x+INFINITESIMAL_DOUBLE);
            else
                x = round(x+0.5+INFINITESIMAL_DOUBLE) - 0.5;
        }
    #endif
    
    bool isCoordsEqual(const myNode& n) const {
        #if GRAPH_TYPE == 8 // COORD_TYPE int
        return ((x==n.x) && (y==n.y));
        #else
        return (fabs(x-n.x)<INFINITESIMAL_DOUBLE  &&  fabs(y-n.y)<INFINITESIMAL_DOUBLE);
        #endif
    }
    
    // *** This must be defined for the node
    bool operator==(const myNode& n) const {
        if (!isCoordsEqual(n)) return (false);
        if (h.size()!=n.h.size()) return (false);
	    for (int a=0; a<h.size(); a++)
	        if (h[a]!=n.h[a]) return (false);
        return (true);
    }
    
    // constructor
    myNode () { }
    myNode (COORD_TYPE xx, COORD_TYPE yy) : x(xx), y(yy) { put_in_grid(); }
    
    
    // Inherited functions being overwritten
    int getHashBin (void) const {
        #if GRAPH_TYPE == 8 // COORD_TYPE int
        return (abs(x));
        #else
        return ( MAX(round(fabs(x)+INFINITESIMAL_DOUBLE), round(fabs(x)-INFINITESIMAL_DOUBLE)) );
        #endif
    }
    
    // print
    void print (std::string head="", std::string tail="\n") const {
        _dosl_cout << _GREEN + head << " (" << this << ")" GREEN_ " x=" << x << ", y=" << y << "; ";
        (g_score==std::numeric_limits<double>::max())? printf("INF") : printf("g_score = %0.8f; h = [", g_score);
        for (int a=0; a<h.size(); ++a){
            printf("%d", a);
            if (a!=h.size()-1) printf(", ");
        } 
        printf("]\n");
        _dosl_cout << "successors: ";
        for (auto it=successors.begin(); it!=successors.end(); ++it)
            printf ("%x (%f), ", it->first, it->second);
        std::cout << tail << _dosl_endl;
    }
    
    // ---------------------------------------------------------------
    // Operator overloading for convex combination -- for Path Recnstruction in SStar algorithm
    
    myNode operator+(const myNode &b) const { // n1 + n2
        myNode ret = (*this);
        ret.x += b.x;
        ret.y += b.y;
        return (ret);
    }
    
    myNode operator*(const double &c) const { // n1 * c (right scalar multiplication)
        myNode ret = (*this);
        ret.x *= c;
        ret.y *= c;
        return (ret);
    } 
};


class CompareByCoordOnly { // functor for invoking isCoordsEqual
public:
    bool operator() (const myNode& n1, const myNode& n2) { return n1.isCoordsEqual(n2); }
} compare_by_coord_only;


// ==============================================================================

class searchProblem : public _DOSL_ALGORITHM::Algorithm<searchProblem,myNode,double>
{
public:
    // Fime names and JSON objects
    std::string   map_image_fName, expt_fName, expt_folderName, expt_Name, out_folderName;
    RSJresource expt_container;
    cvParseMap2d  my_map;
    int nClassesToFind;
    
    // Image display variables / parameters
    cv::Mat image_to_display;
    double PLOT_SCALE;
    double VERTEX_SIZE, LINE_THICKNESS;
    
    // variables for saving image
    int frameno;
    std::ostringstream imgPrefix;
    
    // variables decsribing problem
    COORD_TYPE MAX_X, MIN_X, MAX_Y, MIN_Y;
    myNode startNode, goalNode, lastExpanded;
    
    // homotopy classes
    int nClasses;
    std::vector<myNode> homotopyGoals;
    
    // -----------------------------------------------------------
    
    template<typename T>
    cv::Point cv_plot_coord(T x, T y) {
        return cv::Point(round(PLOT_SCALE*(x-MIN_X)), round(PLOT_SCALE*(y-MIN_Y)));
    }
    
    void cvPlotPoint (cv::Point pt, cv::Scalar colr, int size=1) {
        for (int i=pt.x; i<=pt.x+(size-1); i++)
            for (int j=pt.y; j<=pt.y+(size-1); j++) {
                if (i<0 || i>=image_to_display.cols || j<0 || j>=image_to_display.rows) continue;
                image_to_display.at<cv::Vec3b>(j,i) = cv::Vec3b ((uchar)colr[0], (uchar)colr[1], (uchar)colr[2]);
            }
    }
    
    // Constructor
    searchProblem (std::string expt_f_name, std::string expt_name, std::string out_folder_name)
    {
        expt_fName = expt_f_name; expt_Name = expt_name;
        out_folderName = out_folder_name;
        expt_folderName = expt_fName.substr(0, expt_fName.find_last_of("/\\")+1);
        
        // Read from file
        std::ifstream my_fstream (expt_fName);
        expt_container = RSJresource (my_fstream)[expt_Name];
        
        // obstacle map
        if (expt_container["environment"]["pixmap"].exists()) {
            map_image_fName = expt_folderName + expt_container["environment"]["pixmap"].as<std::string>();
            my_map = cvParseMap2d (map_image_fName, true);
            if (my_map.map.empty()) {
                std::cerr << "Failed to load map image: '" << map_image_fName << "'." << std::endl;
                std::cerr << "Tip: run from the 'examples-dosl' folder, or pass an absolute JSON path as argv[1]." << std::endl;
                std::exit(1);
            }
        }
        else if (expt_container["environment"]["width"].exists() && expt_container["environment"]["height"].exists()) {
            my_map = cvParseMap2d (cv::Mat (expt_container["environment"]["height"].as<int>(), 
                                            expt_container["environment"]["width"].as<int>(),
                                                            CV_8UC3, cv::Scalar(255,255,255)), true);
        }
        
        // read data for planning
        MAX_X=my_map.width(); MIN_X=0; MAX_Y=my_map.height(); MIN_Y=0;
        startNode = myNode (expt_container["start"][0].as<int>(), expt_container["start"][1].as<int>());
        goalNode = myNode (expt_container["goal"][0].as<int>(), expt_container["goal"][1].as<int>());
        startNode.put_in_grid(); goalNode.put_in_grid(); 
        nClassesToFind = expt_container["top_class"].as<int>(2);
        nClasses = 0;
        
        // display options
        PLOT_SCALE = expt_container["plot_options"]["plot_scale"].as<double>(1.0);
        VERTEX_SIZE = expt_container["plot_options"]["vertex_size"].as<double>(1.0);
        LINE_THICKNESS = expt_container["plot_options"]["line_thickness"].as<double>(2.0); // CV_FILLED
        
        // saving options
        frameno = 0;
        imgPrefix << MAKESTR(_DOSL_ALGORITHM) << GRAPH_TYPE << "_homotopy2d_PathPlanning_";
        
        #if _VIS
        image_to_display = my_map.getCvMat (COLOR_MAP);
        cv::resize (image_to_display, image_to_display, cv::Size(), PLOT_SCALE , PLOT_SCALE );
        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display window", image_to_display);
        cv::waitKey(0);
        #endif
        
        // Set planner variables
        all_nodes_set_p->reserve (ceil(MAX_X - MIN_X + 1));
    }
    
    // -----------------------------------------------------------
    
    bool isNodeInWorkspace (const myNode& tn) {
        if ( tn.x<MIN_X || tn.x>=MAX_X || tn.y<MIN_Y || tn.y>=MAX_Y )  return (false);
        return (true);
    }
    
    bool isEdgeAccessible (const myNode& tn1, const myNode& tn2) {
        if ( (!isNodeInWorkspace(tn1)) || !(isNodeInWorkspace(tn2)) )  return (false);
        
        #if GRAPH_TYPE == 8 // COORD_TYPE int
        // the following works only for 8-connected grid!!
        if (tn1.x!=tn2.x && tn1.y!=tn2.y) // diagonal edge
            return ( my_map.isFree ( round(MIN(tn1.x,tn2.x)), round(MIN(tn1.y,tn2.y)) ) );
        else if (tn1.x==tn2.x) // need to check two cells
            return (!( (tn1.x==MAX_X || my_map.isObstacle ( round(tn1.x), round(MIN(tn1.y,tn2.y)) ) ) &
                        ( tn1.x==MIN_X || my_map.isObstacle ( round(tn1.x-1), round(MIN(tn1.y,tn2.y)) ) ) ));
        else if (tn1.y==tn2.y) // need to check two cells
            return (!( ( tn1.y==MAX_Y || my_map.isObstacle ( round(MIN(tn1.x,tn2.x)), round(tn1.y) ) ) &
                        ( tn1.y==MIN_Y || my_map.isObstacle ( round(MIN(tn1.x,tn2.x)), round(tn1.y-1) ) ) ));
        #else
        // TODO: make more precise
        return ( my_map.isFree(approx_floor(tn1.x),approx_floor(tn1.y))  && 
                    my_map.isFree(approx_floor(tn2.x),approx_floor(tn2.y)) );
        #endif
        
        return (true);
    }
    
    // -----------------------------------------------------------
    
    // -----------------------------------------------------------
    // Homotopy signature is now a *reduced word* over wall generators.
    // A generator is an integer encoding:
    //     gen = sign * (2 * componentLabel + orient + 1)
    // where:
    //     componentLabel = obsLabelMap.at<uchar>(blockedCell)  (>=1, set by
    //         cvParseMap2d::computeRepresentativePoints via floodFill)
    //     orient = 0 for a vertical wall (dx was flipped)
    //              1 for a horizontal wall (dy was flipped)
    //     sign   = +1 if the blocked cell lies to the +x/+y side of n,
    //              -1 otherwise (records which face of the wall was hit)
    // Consecutive identical generators cancel (retracing the same bounce),
    // just like the old obstacle-ray word reduction.
    //
    // A single step can reflect off at most one wall; diagonal "corner" hits
    // (8-connected) that would flip both dx and dy are rejected, because
    // the reflection direction there is ambiguous and would correspond to
    // two simultaneous generators.
    // -----------------------------------------------------------

    // Encode a wall-hit as a reduced-word generator and append to tn.h with
    // cancellation against the last generator (rejects consecutive bounces
    // on the same wall, mirroring the old obstacle-ray behaviour).
    void appendWallGenerator (myNode &tn, int componentLabel, int orient, int sign) {
        int gen = sign * (2 * componentLabel + orient + 1);
        if (!tn.h.empty() && tn.h.back() == -gen) {
            // opposite-sign same wall => cancel
            tn.h.pop_back();
        } else if (!tn.h.empty() && tn.h.back() == gen) {
            // same-wall consecutive bounce => reject by cancellation
            // (this is the "no consecutive ray crossing" rule, transplanted)
            tn.h.pop_back();
        } else {
            tn.h.push_back(gen);
        }
    }

    // Read the obstacle-component label at an integer cell. Returns 0 for
    // free cells or out-of-bounds, >=1 for obstacle components.
    int obstacleComponentAt (int cx, int cy) {
        if (cx < 0 || cy < 0 || cx >= my_map.width() || cy >= my_map.height())
            return 1; // treat the outer frame as a single "boundary" component
        if (my_map.isFree(cx, cy)) return 0;
        if (!my_map.obsLabelMap.empty())
            return (int) my_map.obsLabelMap.at<uchar>(cy, cx);
        return 1; // fallback: one generic obstacle component
    }

    // Given a parent n and an attempted straight step to (n.x+dx, n.y+dy),
    // produce either the straight successor or a reflected successor.
    // Returns true on success and fills *tnOut / *costOut / updates tnOut->h
    // (which is initialised from n.h on entry). Returns false if neither the
    // straight step nor a valid reflection is available.
    bool buildSuccessor (myNode &n, double dx, double dy,
                         myNode *tnOut, double *costOut)
    {
        tnOut->h = n.h;

        // --- straight step ---
        myNode straight; straight.x = n.x + dx; straight.y = n.y + dy;
        straight.put_in_grid();
        if (isEdgeAccessible(n, straight)) {
            tnOut->x = straight.x;
            tnOut->y = straight.y;
            tnOut->put_in_grid();
            *costOut = sqrt(dx*dx + dy*dy);
            return true;
        }

        // --- blocked: decide which wall was hit ---
        // Inspect the cell the straight step tried to land on.
        int bx = approx_floor(straight.x);
        int by = approx_floor(straight.y);

        // Must stay in the workspace for a reflection to be meaningful.
        if (!isNodeInWorkspace(straight) && !(bx < 0 || by < 0
                                           || bx >= my_map.width() || by >= my_map.height()))
            return false;

        // Decompose the displacement into which axis actually crossed into
        // an obstacle cell. We probe the neighbours of n: if (n+dx, n.y) is
        // obstructed we have a vertical wall; if (n.x, n+dy) is obstructed
        // we have a horizontal wall.
        int nix = approx_floor(n.x), niy = approx_floor(n.y);

        bool vWall = false, hWall = false;
        int vCompLabel = 0, hCompLabel = 0;

        if (fabs(dx) > INFINITESIMAL_DOUBLE) {
            int probeX = approx_floor(n.x + dx);
            int l = obstacleComponentAt(probeX, niy);
            if (l > 0) { vWall = true; vCompLabel = l; }
        }
        if (fabs(dy) > INFINITESIMAL_DOUBLE) {
            int probeY = approx_floor(n.y + dy);
            int l = obstacleComponentAt(nix, probeY);
            if (l > 0) { hWall = true; hCompLabel = l; }
        }

        // If neither axis-probe identified a wall, the blocked cell is a
        // pure diagonal (corner) obstruction -- use the diagonal cell's
        // component as a corner. We reject corners: two generators at once
        // would require a combined encoding and reflection is ambiguous.
        if (!vWall && !hWall)
            return false;

        // Reject simultaneous vertical+horizontal hits (concave corner):
        // ambiguous reflection direction, drop this successor.
        if (vWall && hWall)
            return false;

        double rdx = dx, rdy = dy;
        int orient, sign, compLabel;
        if (vWall) {
            rdx = -dx;                       // flip x-component
            orient = 0;                      // vertical wall
            sign   = (dx > 0) ? +1 : -1;     // which face of wall was hit
            compLabel = vCompLabel;
        } else { // hWall
            rdy = -dy;                       // flip y-component
            orient = 1;                      // horizontal wall
            sign   = (dy > 0) ? +1 : -1;
            compLabel = hCompLabel;
        }

        myNode reflected; reflected.x = n.x + rdx; reflected.y = n.y + rdy;
        reflected.put_in_grid();

        // The reflected cell must itself be a legal edge from n.
        if (!isEdgeAccessible(n, reflected))
            return false;

        // Accept: commit the successor and update the signature word.
        tnOut->x = reflected.x;
        tnOut->y = reflected.y;
        tnOut->put_in_grid();
        appendWallGenerator(*tnOut, compLabel, orient, sign);

        // Reflected edge length equals incident edge length; cost is the
        // straight-step distance (we are "bouncing" in place of travelling
        // through the wall).
        *costOut = sqrt(dx*dx + dy*dy);
        return true;
    }

    void getSuccessors (myNode &n, std::vector<myNode>* s, std::vector<double>* c) // *** This must be defined
    {
        // This function should account for obstacles and size of environment.
        myNode tn;
        double cst;

        #if GRAPH_TYPE == 8
        for (int a=-1; a<=1; ++a)
            for (int b=-1; b<=1; ++b) {
                if (a==0 && b==0) continue;

                #ifdef DOSL_ALGORITHM_SStar
                int xParity = ((int)round(fabs(n.x))) % 2;
                if (xParity==0 && (a!=0 && b==-1)) continue;
                if (xParity==1 && (a!=0 && b==1)) continue;
                #endif

                if (!buildSuccessor(n, (double)a, (double)b, &tn, &cst)) continue;

                s->push_back(tn);
                c->push_back(cst);
            }

        #elif GRAPH_TYPE == 6
        double th, dx, dy;
        for (int a=0; a<6; ++a) {
            th = a * PI_BY_3;
            dx = 1.0*cos(th);
            dy = 1.0*sin(th);

            if (!buildSuccessor(n, dx, dy, &tn, &cst)) continue;

            s->push_back(tn);
            c->push_back(cst);
        }

        #endif
    }
    
    // -----------------------------------------------------------
    
    double getHeuristics (myNode& n)
    {
        /* double dx = goalNode.x - n.x;
        double dy = goalNode.y - n.y;
        return (sqrt(dx*dx + dy*dy)); */
        return (0.0);
    }
    
    // -----------------------------------------------------------
    
    std::vector<myNode> getStartNodes (void) 
    {
        std::vector<myNode> startNodes;
        
        startNodes.push_back (startNode);
        #if _VIS
        cv::circle (image_to_display, cv_plot_coord(startNode.x,startNode.y), VERTEX_SIZE*PLOT_SCALE, 
                                                                cv::Scalar (200.0, 150.0, 150.0), -1, 8);
        #endif
        
        return (startNodes);
    }
    
    // -----------------------------------------------------------
    
    void nodeEvent (myNode &n, unsigned int e) 
    {
        #if _VIS
        
        cv::Scalar col = cv::Scalar(0.0, 0.0, 0.0);;
        int thickness = -1; //lineThickness;
        double radFactor = 1.0;
        
        bool pauseForVis=false, drawVertex=true;
        
        // --------------------------------------------
        if (e & EXPANDED) {
            lastExpanded = n;
            bool cameFromNull = false;
            #ifdef DOSL_ALGORITHM_SStar
            cameFromNull = (n.CameFromSimplex==NULL);
            #endif
            if (!cameFromNull) {
                #if VERTEX_COLORS
                std::vector<myNode*> nodes_at_same_xy = all_nodes_set_p->getall (n, compare_by_coord_only);
                double rb_intensity = MAX(0.0, 255.0 - 50.0*nodes_at_same_xy.size());
                col = cv::Scalar (rb_intensity, 255.0, rb_intensity);
                #else
                col = cv::Scalar(255.0, 255.0, 255.0);
                #endif
            }
            else {
                printf ("expanded, but came-from is NULL!!\n");
                // pauseForVis = true;
            }
        }
        
        else if ((e & HEAP) == PUSHED) {
            #if VERTEX_COLORS
            col = cv::Scalar(255.0, 0.0, 0.0); // blue
            #else
            col = cv::Scalar(200.0, 200.0, 200.0);
            #endif
        }
        
        else if (e & UNEXPANDED) {
            col = cv::Scalar(255.0, 0.0, 150.0); // purple
            printf ("Backtrack started!!\n");
        }
        
        else
            return;
                
        //-------------------------------------------
        if (drawVertex) {
            double nodeRad = radFactor*VERTEX_SIZE;
            if (n.x<MAX_X  &&  n.y<MAX_Y  &&  my_map.isFree(round(n.x), round(n.y)) )
                cvPlotPoint (cv_plot_coord(n.x,n.y), col, PLOT_SCALE);
        }
        
        if (expand_count%VIS_INTERVAL == 0  ||  node_heap_p->size() == 0) {
            cv::imshow("Display window", image_to_display);
            std::cout << std::flush;
            if (SAVE_IMG_INTERVAL>0 && expand_count%SAVE_IMG_INTERVAL == 0) {
                char imgFname[1024];
                sprintf(imgFname, "%s%s%05d.png", out_folderName.c_str(), imgPrefix.str().c_str(), expand_count);
                cv::imwrite(imgFname, image_to_display);
            }
            cv::waitKey(1); //(10);
        }
        
        if (pauseForVis) {
            cv::waitKey();
        }
        #endif
    }
    
    // ---------------------------------------
    
    bool stopSearch (myNode &n) {
        if (n.isCoordsEqual(goalNode)) {
            homotopyGoals.push_back (n);
            ++nClasses;
            n.print ("Found a path to ");
            if (nClasses>=nClassesToFind)
                return (true);
        }
        return (false);
    }
};

// ==============================================================================

int main(int argc, char *argv[])
{
    //RUNTIME_VERBOSE_SWITCH = 0;
    compute_program_path();
    
    // std::string expt_f_name = program_path+"../files/expt/basic_experiments.json", expt_name="L457_expt1";
    std::string expt_f_name = program_path+"../files/expt/basic_experiments.json", expt_name="simple_expt1";

    if (argc == 2) {
        expt_name = argv[1];
    }
    else if (argc > 2) {
        expt_f_name = argv[1];
        expt_name = argv[2];
    }
    
    printf (_BOLD _YELLOW "Note: " YELLOW_ BOLD_ "Using algorithm " _YELLOW  MAKESTR(_DOSL_ALGORITHM)  YELLOW_ 
                ". Run 'make' to recompile with a different algorithm.\n"
                "Note: This program currently does not support 'ThetaStar' algorithm.\n");
    
    searchProblem test_search_problem (expt_f_name, expt_name, program_path+"../files/out/");
    test_search_problem.search();
    
    // -------------------------------------
    #if _STAT
    char statFname[1024];
    sprintf(statFname, "%s%s_%s.txt", test_search_problem.out_folderName.c_str(), 
                            test_search_problem.imgPrefix.str().c_str(), test_search_problem.map_image_fName.c_str());
    FILE* pFile;
    pFile = fopen (statFname,"a+");
    if (pFile!=NULL)
        fprintf (pFile, "%s:\n", test_search_problem.expt_Name.c_str());
    #endif
    
    for (int i=0; i<test_search_problem.homotopyGoals.size(); ++i) {
        // get path
        auto path = test_search_problem.reconstruct_weighted_pointer_path (test_search_problem.homotopyGoals[i]);
        double cost = 0.0;
        
        // Print and draw path
        #if _VIS
        //printf("\nPath: ");
        #endif
        myNode thisPt=test_search_problem.startNode, lastPt;
        std::vector<myNode> allPts;
        for (int a=path.size()-1; a>=0; --a) {
            lastPt = thisPt;
            thisPt = myNode(0.0, 0.0);
            for (auto it=path[a].begin(); it!=path[a].end(); ++it) {
                thisPt.x += it->second * it->first->x;
                thisPt.y += it->second * it->first->y;
            }
            allPts.push_back (thisPt);
            #if _VIS
            //printf ("[%f,%f]; ", (double)thisPt.x, (double)thisPt.y);
            cv::line (test_search_problem.image_to_display, 
                    test_search_problem.cv_plot_coord(thisPt.x,thisPt.y), test_search_problem.cv_plot_coord(lastPt.x,lastPt.y), 
                                cv::Scalar(200.0,100.0,100.0),
                                        test_search_problem.LINE_THICKNESS*test_search_problem.PLOT_SCALE);
            #endif
            cost += sqrt ((thisPt.x-lastPt.x)*(thisPt.x-lastPt.x) + (thisPt.y-lastPt.y)*(thisPt.y-lastPt.y));
        }
        
        #if _VIS
        //printf ("\n");
        for (int a=allPts.size()-1; a>=0; --a)
            cv::circle (test_search_problem.image_to_display, 
                    test_search_problem.cv_plot_coord(allPts[a].x,allPts[a].y),
                        test_search_problem.VERTEX_SIZE*test_search_problem.PLOT_SCALE, cv::Scalar(150.0,0.0,255.0), -1, 8);
        #endif
        
        printf ("\tclass: %d, cost: %f.\n", i, cost);
        #if _STAT
        if (pFile!=NULL)
            fprintf (pFile, "\tclass: %d, cost: %f.\n", i, cost);
        #endif
    }
    
    #if _VIS
    cv::imshow("Display window", test_search_problem.image_to_display);
    #endif
    
    #if _STAT
    if (pFile!=NULL)
        fclose (pFile);
    #endif
    
    test_search_problem.clear();
    
    #if _VIS
    if (SAVE_IMG_INTERVAL != 0) {
        char imgFname[1024];
        sprintf(imgFname, "%s%s_path.png", test_search_problem.out_folderName.c_str(), 
                                                test_search_problem.imgPrefix.str().c_str());
        cv::imwrite(imgFname, test_search_problem.image_to_display);
    }
    //printf ("\ncomputation time = %f\n", 0.0);
    cv::waitKey();
    #endif
}