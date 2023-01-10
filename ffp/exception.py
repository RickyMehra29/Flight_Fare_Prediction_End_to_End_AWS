import sys, os

def error_message_details(exception_error, sys):

    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured at python script name [{0}] at line number [{1}] with error message as [{2}]" .format(
                    file_name, exc_tb.tb_lineno, str(exception_error)) 
    return error_message
    

class FlightFareException(Exception):
    
    def __init__(self, error_message, sys):
        self.error_message = error_message_details(error_message, sys=sys)    

    def __str__(self):
        return self.error_message