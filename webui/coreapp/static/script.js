$(document).ready( function() {
        $("#image_file").change(function() {
            console.log("JS upload")
            $("#formUpload").submit();
            setTimeout(15000);
        });

});