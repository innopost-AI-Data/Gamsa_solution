$('#evaluate_csv').on('click', function () {
    var url = $('#fileForm').attr('action');
    var form_data = new FormData();
    console.log($('#file').prop('files')[0]);
    form_data.append('file', $('#file').prop('files')[0]);

    $.ajax({
        type: 'POST',
        url: url,
        data: form_data,
        contentType: false,
        processData: false,
        success: function (data) {
            $("#csv-result").prepend(data.result);
        }
    });
});

$(document).on('click', 'button#download_btn', function () {
    filename = $(event.target).attr('name')
    var html = $(event.target).parent().prev().html()
    var link = document.createElement('a');
    link.setAttribute('download', filename + '.txt');
    link.setAttribute('href', 'data:application/txt' + ';charset=utf-8,' + encodeURIComponent(html));
    link.click();
});

$('#evaluate_csv').on('click', function () {
    var url = $('#fileForm').attr('action');
    var form_data = new FormData();
    console.log($('#file').prop('files')[0]);
    form_data.append('file', $('#file').prop('files')[0]);

    $.ajax({
        type: 'POST',
        url: url,
        data: form_data,
        contentType: false,
        processData: false,
        success: function (data) {
            $("#csv-result").prepend(data.result);
        }
    });
});
