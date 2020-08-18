request = function()
    local headers = {}
    headers["Content-Type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"

    path = "/api/obj_det/predict"

    body = '------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n'
    body = body .. [[Content-Disposition: form-data; name="file"; filename="dog.jpg"]]
    body = body .. '\r\n\r\n'
    file = io.open("dog.jpg", "rb")
    body = body .. file:read("*a")
    body = body .. '\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--'
    io.close(file)
    return wrk.format('POST', path, headers, body)
end
