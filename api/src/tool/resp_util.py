def get_home_response():

    content = """
                <body>
                <form action="/api/obj_det/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" multiple>
                <input type="submit">
                </form>

                </body>
            """
    return content

def get_prediction_response(out_path):
    
    content = """
                <body>
                <form action="/api/obj_det/predict" enctype="multipart/form-data" method="post">
                <input name="file" type="file" multiple>
                <input type="submit">
                </form>

                <img src='"""
    content = content+out_path
    trail = """
                '>
                </body>
                """
    content = content+trail
    return content
