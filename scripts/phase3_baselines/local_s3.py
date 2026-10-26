"""A Java S3Client that serves getObject from a local folder (key = path under it), for replays of synthetic files."""
from pathlib import Path

import jpype
from jpype import JImplements, JOverride


def local_s3_client(root: Path):
    Proxy = jpype.JClass("java.lang.reflect.Proxy")
    S3Client = jpype.JClass("software.amazon.awssdk.services.s3.S3Client")
    GetObjectRequest = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectRequest")
    GetObjectResponse = jpype.JClass("software.amazon.awssdk.services.s3.model.GetObjectResponse")
    NoSuchKeyException = jpype.JClass("software.amazon.awssdk.services.s3.model.NoSuchKeyException")
    ResponseBytes = jpype.JClass("software.amazon.awssdk.core.ResponseBytes")
    ResponseInputStream = jpype.JClass("software.amazon.awssdk.core.ResponseInputStream")
    AbortableInputStream = jpype.JClass("software.amazon.awssdk.http.AbortableInputStream")
    ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")

    @JImplements("java.lang.reflect.InvocationHandler")
    class _Handler:
        @JOverride
        def invoke(self, proxy, method, args):
            name = str(method.getName())
            if name in ("getObject", "getObjectAsBytes"):
                request = args[0]
                if not isinstance(request, GetObjectRequest):
                    builder = GetObjectRequest.builder()
                    request.accept(builder)
                    request = builder.build()
                path = root / str(request.key())
                if not path.is_file():
                    raise NoSuchKeyException.builder().message(f"no local file for {request.key()}").build()
                data = jpype.JArray(jpype.JByte)(path.read_bytes())
                response = GetObjectResponse.builder().contentLength(jpype.JLong(len(data))).build()
                if name == "getObjectAsBytes":
                    return ResponseBytes.fromByteArray(response, data)
                return ResponseInputStream(response, AbortableInputStream.create(ByteArrayInputStream(data)))
            if name == "close":
                return None
            if name == "serviceName":
                return "s3"
            if name == "toString":
                return f"LocalS3({root})"
            if name == "hashCode":
                return jpype.JInt(id(self) & 0x7FFFFFFF)
            if name == "equals":
                return proxy is args[0]
            raise jpype.JClass("java.lang.UnsupportedOperationException")(f"LocalS3 does not support {name}")

    return Proxy.newProxyInstance(S3Client.class_.getClassLoader(), [S3Client.class_], _Handler())
