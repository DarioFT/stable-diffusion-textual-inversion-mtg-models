# pip install html5print requests huggingface-hub Pillow

import argparse
import datetime
import os
import shutil
import sys
from urllib import request as ulreq

import requests
from huggingface_hub import HfApi
from PIL import ImageFile

parser = argparse.ArgumentParser()
parser.add_argument('out_file', nargs='?', help='file to save to', default='stable-diffusion-textual-inversion-models.html')
args = parser.parse_args()

print('Will save to file:', args.out_file)

# Init some stuff before saving the time
api = HfApi()
models_list = ["nissa-revane", "tamiyo", "chandra-nalaar", "kiora", "vraska", "elspeth-tirel", "kaya-ghost-assasin", 
"nahiri", "saheeli-rai", "huatli", "vivien-reid", "liliana-vess", "ashiok", "teferi", "dovin-baan"]
models_list.sort()

# Save the time now before we do the hard work
dt = datetime.datetime.now()
tz = dt.astimezone().tzname()

html_struct = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  
  <!-- HTML Meta Tags -->
  <meta charset="utf-8">
  <title>MTG - Stable Diffusion Textual Inversion Embeddings</title>
  <meta name="description" content="Curated list of custom HuggingFace textual inversion library Magic: The Gathering models for download.">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <!-- Facebook Meta Tags -->
  <meta property="og:url" content="https://darioft.github.io/stable-diffusion-textual-inversion-mtg-models/">
  <meta property="og:type" content="website">
  <meta property="og:title" content="Magic: The Gathering - Stable Diffusion Textual Inversion Embeddings">
  <meta property="og:description" content="Curated list of custom HuggingFace textual inversion library Magic: The Gathering models for download.">
  <meta property="og:image" content="https://darioft.github.io/stable-diffusion-textual-inversion-mtg-models/og.jpg">

  <!-- Twitter Meta Tags -->
  <meta name="twitter:card" content="summary_large_image">
  <meta property="twitter:domain" content="darioft.github.io">
  <meta property="twitter:url" content="https://darioft.github.io/stable-diffusion-textual-inversion-mtg-models/">
  <meta name="twitter:title" content="Magic: The Gathering - Stable Diffusion Textual Inversion Embeddings">
  <meta name="twitter:description" content="Curated list of custom HuggingFace textual inversion library Magic: The Gathering models for download.">
  <meta name="twitter:image" content="https://darioft.github.io/stable-diffusion-textual-inversion-mtg-models/og.jpg">


  <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-iYQeCzEYFbKjA/T2uDLTpkwGzCiq6soy8tYaI1GyVh/UjpbCx/TYkiZhlZB6+fzT" crossorigin="anonymous">

  <link rel="apple-touch-icon" sizes="180x180" href="/stable-diffusion-textual-inversion-models/apple-touch-icon.png">
  <link rel="icon" type="image/png" sizes="32x32" href="/stable-diffusion-textual-inversion-models/favicon-32x32.png">
  <link rel="icon" type="image/png" sizes="16x16" href="/stable-diffusion-textual-inversion-models/favicon-16x16.png">
  <link rel="manifest" href="/stable-diffusion-textual-inversion-models/site.webmanifest">
  <link rel="mask-icon" href="/stable-diffusion-textual-inversion-models/safari-pinned-tab.svg" color="#ee9321">
  <link rel="shortcut icon" href="favicon.ico">
  <meta name="msapplication-TileColor" content="#ee9321">
  <meta name="msapplication-config" content="/stable-diffusion-textual-inversion-models/browserconfig.xml">
  <meta name="theme-color" content="#ee9321">

</head>

<body>
  <style>
    .thumbnail {{
        max-width: 185px;
        display: block;
        padding-top: 5px;
        padding-bottom: 5px;
      }}
      
    .img-max {{
        max-width: 500px;
        width:100%;
      }}

    .model-title {{
        margin-top: 40px;
      }}

    body {{
        background-color: #0000ff0d !important;
    }}

    .model-title > a {{
        color: initial !important;
        text-decoration: none !important;
    }}
  </style>

  <div class="container">
    <div class="jumbotron text-center" style="margin-top: 45px;margin-right: 45px;margin-bottom: 0px;margin-left: 45px;">
      <img src="mtglogo.png" class="img-fluid img-max">
      <h1>Stable Diffusion Textual Inversion Embeddings</h1>
    </div>
    <div style="text-align: center;margin-bottom: 45px;font-size: 8pt;">
      <p>
        <div>
            <a href="https://github.com/DarioFT/stable-diffusion-textual-inversion-mtg-models/actions/workflows/generate_static_html.yml"><img src="https://github.com/DarioFT/stable-diffusion-textual-inversion-mtg-models/actions/workflows/generate_static_html.yml/badge.svg"></a>
        </div>
        <br>
        <i>Page updated on <a class="btn-link" style="cursor: pointer;text-decoration: none;" data-toggle="tooltip" data-placement="bottom" title="{dt.strftime(f"%m-%d-%Y %H:%M:%S {tz}")}">{dt.strftime("%A %B %d, %Y")}</a>.</i>
      </p>
    </div>

    <p>
      Curated list of <a href="https://huggingface.co/sd-concepts-library">HuggingFace textual inversion library</a> Magic: The Gathering models. 
      <br>
      There are currently {len(models_list)} textual inversion embeddings listed. These are meant to be used with <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">AUTOMATIC1111's SD WebUI</a>.
    </p>

    <p>
      Embeddings are downloaded straight from the HuggingFace repositories. <br> Want to quickly test concepts? 
      Try the <a href="https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer">Stable Diffusion Conceptualizer</a> on HuggingFace. <a href="https://huggingface.co/docs/diffusers/main/en/training/text_inversion">More info on textual inversion.</a>
    </p>

    <hr>
"""

i = 1
for model_name in models_list:
    # For testing
    # if i == 4:
    #     break

    print(f'{i}/{len(models_list)} -> {model_name}')

    html_struct = html_struct + f'<div><h3 class="model-title" id="{model_name}"><a href="#{model_name}">{model_name.replace("-", " " ).title()}</a></h3>'

    # Get the concept images from the huggingface repo
    restricted = False
    try:
        files = api.list_repo_files(
            repo_id=f'sd-concepts-library/{model_name}')
        concept_images = [i for i in files if i.startswith('concept_images/')]
        sample_images = [i for i in files if i.startswith('sample_images/')]
    except requests.exceptions.HTTPError:
        # Sometimes an author will require you to share your contact info to gain access
        restricted = True

    if restricted:
        html_struct = html_struct + f"""
<p>
  {model_name} is restricted and you must share your contact information to view this repository.
  <a type="button" class="btn btn-link" href="https://huggingface.co/sd-concepts-library/{model_name}/">View Repository</a>
</p>
        """
    else:
        html_struct = html_struct + f"""
<p>
  <button type="button" class="btn btn-primary" onclick="downloadAs('https://huggingface.co/sd-concepts-library/{model_name}/resolve/main/learned_embeds.bin', '{model_name.replace("-", " " )}.pt')">Download {model_name.replace("-", " " )}.pt</button>
  <a type="button" class="btn btn-link" href="https://huggingface.co/sd-concepts-library/{model_name}/">View Repository</a>
</p>

<ul class="nav nav-tabs" id="{model_name}-tab" role="tablist">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="{model_name}-sample-tab" data-bs-toggle="tab" data-bs-target="#{model_name}-sample" type="button" role="tab" aria-controls="{model_name}-sample" aria-selected="true">Samples</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="{model_name}-model-tab" data-bs-toggle="tab" data-bs-target="#{model_name}-model" type="button" role="tab" aria-controls="{model_name}-model" aria-selected="false">Model</button>
  </li>
</ul>
<div class="tab-content" id="{model_name}-tab">
    <div class="tab-pane fade show active" id="{model_name}-sample" role="tabpanel" aria-labelledby="{model_name}-sample-tab">
        <div class="row">
        """

        img_count = 4
        if len(sample_images) < 4:
            img_count = len(sample_images)

        for x in range(img_count):
            html_struct = html_struct + f"""
        <div class="col-sm">
          <img class="thumbnail mx-auto img-fluid" loading="lazy" src="https://huggingface.co/sd-concepts-library/{model_name}/resolve/main/{sample_images[x]}">
        </div>
            """
        html_struct = html_struct + '</div></div>'

    html_struct = html_struct + f"""


    <div class="tab-pane fade" id="{model_name}-model" role="tabpanel" aria-labelledby="{model_name}-model-tab">
        <div class="row">

    """

    img_count = 4
    if len(concept_images) < 4:
        img_count = len(concept_images)

    for x in range(img_count):
        html_struct = html_struct + f"""

    <div class="col-sm">
      <img class="thumbnail mx-auto img-fluid" loading="lazy" src="https://huggingface.co/sd-concepts-library/{model_name}/resolve/main/{concept_images[x]}">
    </div>
                """

    html_struct = html_struct + '</div></div>'
    i = i + 1

html_struct = html_struct + """

</div>

  </div>
  
  <div class="container">
  <footer class="py-3 my-4">
    <div class="nav justify-content-center border-bottom pb-3 mb-3">
    </div>
    <p class="text-center text-muted">Magic: the Gathering is TM and copyright Wizards of the Coast, Inc, a subsidiary of Hasbro, Inc. All rights reserved. All art used for model training is property of their respective artists and/or Wizards of the Coast. This site is not produced, affiliated or endorsed by Wizards of the Coast, Inc.</p>
  </footer>
</div>
  
  <script>
    // Download the file under a different name
    const downloadAs = (url, name) => {
      axios.get(url, {
          headers: {
            "Content-Type": "application/octet-stream"
          },
          responseType: "blob"
        })
        .then(response => {
          const a = document.createElement("a");
          const url = window.URL.createObjectURL(response.data);
          a.href = url;
          a.download = name;
          a.click();
        })
        .catch(err => {
          console.log("error", err);
        });
    };
    document.addEventListener("DOMContentLoaded", () => {
        // Enable tooltips
        $(function() {
          $('[data-toggle="tooltip"]').tooltip({
            placement: "bottom"
          })
        });
   });
  </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-u1OknCvxWvY5kfmNBILK2hRnQC3Pr17a+RTT6rIHI7NnikvbZlHgTPOOmMi466C8" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js" integrity="sha512-bLT0Qm9VnAYZDflyKcBaQ2gg0hSYNQrJ8RilYldYQ1FxQYoCLtUjuuRuZo+fjqhx/qtq/1itJ0C2ejDxltZVFg==" crossorigin="anonymous"></script>
    <!--<script src="/stable-diffusion-textual-inversion-models/jquery.waypoints.min.js"></script>-->
</body>
"""

f = open(args.out_file, 'w', encoding='utf-8')
f.write(html_struct)
f.close()


