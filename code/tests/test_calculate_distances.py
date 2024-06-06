#Author: Kent O'Sullivan // osullik@umd.edu

#Core Imports 
import unittest
import os
import sys

#Library Imports

#User Imports
sys.path.append('..') #append parent directory with the python files to path
from calculate_distances import GeoCalc

# Classes

class Test_Geocalc(unittest.TestCase):

    def setUp(self) -> None:
        self.data_directory = os.path.join('..','..','data','test_data')
        self.gc = GeoCalc()
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_geocalc_object_exists(self):
        self.assertTrue(self.gc)

    # Get Distance Between Two Cities

        # Reverse Geocode City

    def test_reverse_geocode_city(self):

        # Manilla NSW: -30.74747553957213, 150.71987623476613
        input = "Manilla, NSW"
        result = {"name":input,
                  "lat":-30.74,
                  "lon":150.72}

        q_result = self.gc.get_coords_from_name(location=input)

        self.assertAlmostEqual(q_result['lat'], result['lat'], places=1)
        self.assertAlmostEqual(q_result['lon'], result['lon'], places=1)
        
    # Calc distace in KM
    # Calc distance in Miles
    def test_get_distances(self):

        coords_1 = {
                    'name'  :   "Manilla, NSW",
                    "lat"   :   -30.74,
                    "lon"   :   150.72}

        #Tamworth NSW -31.090976550259068, 150.9312896910818
        coords_2 = {"name"  :   "Tamworth, NSW",
                    "lat"   :   -31.09,
                    "lon"   :   150.93}
        
        dist_km = 43.70
        dist_mi = 27.15

        distance_km = self.gc.get_distance_between_coords_in_km(coords_1,coords_2)
        self.assertAlmostEqual(distance_km, dist_km, places=0)

        distance_mi = self.gc.get_distance_between_coords_in_mi(coords_1,coords_2)
        self.assertAlmostEqual(distance_mi, dist_mi, places=2)

    # Get Ratio between Distances

    def test_calc_distance_ratio(self):

        d1_1 = 50
        d1_2 = 50
        r1 = 1.0

        d2_1 = 10
        d2_2 = 100
        r2 = 0.1

        d3_1 = 100
        d3_2 = 10
        r3 = 0.1

        self.assertEqual(self.gc.calc_distance_ratio(d1_1, d1_2), r1)
        self.assertEqual(self.gc.calc_distance_ratio(d2_1, d2_2), r2)
        self.assertEqual(self.gc.calc_distance_ratio(d3_1, d3_2), r3)

    def test_calc_normalized_distance_ratio(self):

        norm_factor = 1000  #In general, use width of country or area being tested

        d1_1 = 50
        d1_2 = 50
        r1 = 0.0

        d2_1 = 10
        d2_2 = 100
        r2 = 0.090

        d3_1 = 1000
        d3_2 = 10
        r3 = 0.99

        self.assertEqual(round(self.gc.calc_normalized_distance_ratio(d1_1, d1_2, norm_factor=norm_factor),ndigits=4), r1)
        self.assertEqual(round(self.gc.calc_normalized_distance_ratio(d2_1, d2_2, norm_factor=norm_factor),ndigits=4), r2)
        self.assertEqual(round(self.gc.calc_normalized_distance_ratio(d3_1, d3_2, norm_factor=norm_factor),ndigits=4), r3)

    def test_calculate_nearness(self):

        norm_factor = 4000 #width of Australia

        loc1a = "Sydney, NSW"          # -33.86916562287265, 151.20647790454234
        loc1b = "Newcastle, NSW"       # -32.92590561017436, 151.78156829309015

        loc2a = "Melbourne, VIC"        # -37.813362331701775, 144.9642827386572
        loc2b = "Geelong, VIC"          # -38.146943834573975, 144.3571800907058

        loc3a = "Sydney,NSW"            # -33.86916562287265, 151.20647790454234    
        loc3b = "Perth, WA"             # -31.95162999189365, 115.85937803062562

        expected_1 = (118.06, 64.75, 0.01,'very_similar')   # Syd New Mel Geel ratio

        expected_2 = (118.06, 3297.20, 0.79,'different') # syd, new, syd, per ratio

        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc2a, loc2b, norm_factor=norm_factor)[0], expected_1[0], places=2)
        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc2a, loc2b, norm_factor=norm_factor)[1], expected_1[1], places=2)
        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc2a, loc2b, norm_factor=norm_factor)[2], expected_1[2], places=2)
        self.assertEqual(self.gc.calculate_nearness(loc1a, loc1b, loc2a, loc2b, norm_factor=norm_factor)[3], expected_1[3])

        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc3a, loc3b, norm_factor=norm_factor)[0], expected_2[0], places=2)
        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc3a, loc3b, norm_factor=norm_factor)[1], expected_2[1], places=2)
        self.assertAlmostEqual(self.gc.calculate_nearness(loc1a, loc1b, loc3a, loc3b, norm_factor=norm_factor)[2], expected_2[2], places=2)
        
        self.assertEqual(self.gc.calculate_nearness(loc1a, loc1b, loc3a, loc3b, norm_factor=norm_factor)[3], expected_2[3])


if __name__=="__main__":
    unittest.main()