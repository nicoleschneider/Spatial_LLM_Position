#Author: Kent O'Sullivan | osullik@umd.edu

# Core Imports

# Library Imports

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# User Imports

# Classes and Functions

class GeoCalc():
    def __init__(self):
        self.geolocator = Nominatim(user_agent="GeoCalc",
                                    timeout=5) #Improves error tolerance by increasing timeout to 5 seconds from default of 1
        pass

    def get_coords_from_name(self, location:str)->dict:

        result = self.geolocator.geocode(location)
        return({"name":location,
                "lat":result.latitude, 
                "lon":result.longitude})

    def get_distance_between_coords_in_km(self, loc_1:dict, loc_2:dict)->float:

        l1 = (loc_1['lat'], loc_1['lon'])
        l2 = (loc_2['lat'], loc_2['lon'])

        dist = geodesic(l1, l2).kilometers

        return(dist)
    
    def get_distance_between_coords_in_mi(self, loc_1:dict, loc_2:dict)->float:

        l1 = (loc_1['lat'], loc_1['lon'])
        l2 = (loc_2['lat'], loc_2['lon'])

        dist = geodesic(l1, l2).miles

        return(dist)
    
    def calc_distance_ratio(self, d1:float, d2:float)->float:

        res = d1/d2

        if (res) >1.0:
            res = d2/d1

        return(res)
    
    def calc_normalized_distance_ratio(self, d1:float, d2:float, norm_factor:float)->float:

        #Norm factor should be roughly diameter of country being analyzed

        r1 = d1/norm_factor
        r2 = d2/norm_factor

        # res = abs((r1-r2))
        res = (r1-r2) #Negative values indicate it is further than example

        return(res)


    def calculate_nearness(self, loc1a:str, loc1b:str, loc2a:str, loc2b:str, norm_factor:int,measure:str='KM')->float:

        #get coords
        c1a = self.get_coords_from_name(location=loc1a)
        c1b = self.get_coords_from_name(location=loc1b)
        c2a = self.get_coords_from_name(location=loc2a)
        c2b = self.get_coords_from_name(location=loc2b)

        if c2b is None:
            print("Unable to locate:", loc2b)
            return ((None, None, None, 'unable_to_locate'))

        #get_distances

        if measure == "KM":
            dist_1 = self.get_distance_between_coords_in_km(loc_1=c1a, loc_2=c1b)
            dist_2 = self.get_distance_between_coords_in_km(loc_1=c2a, loc_2=c2b)
        else:
            dist_1 = self.get_distance_between_coords_in_mi(loc_1=c1a, loc_2=c1b)
            dist_2 = self.get_distance_between_coords_in_mi(loc_1=c2a, loc_2=c2b)

        #Get Ratio

        ratio = self.calc_normalized_distance_ratio(d1=dist_1, d2=dist_2, norm_factor=norm_factor)

        if ratio <= 0.05:
            return((dist_1, dist_2, ratio, 'very_similar'))
        
        if ratio <= 0.10:
            return((dist_1, dist_2, ratio, 'similar'))
        
        if ratio <= 0.25:
            return((dist_1, dist_2, ratio, 'somewhat_similar'))

        return ((dist_1, dist_2, ratio, 'different'))


# Main
