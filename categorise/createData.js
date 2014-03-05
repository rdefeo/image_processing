var mongoClient = require('mongodb').MongoClient;
var fs = require('fs');
var mongoUri = process.env.MONGOHQ_GETTER_URL || "mongodb://localhost/getter";

var copyFileSync = function(srcFile, destFile, encoding) {
  var content = fs.readFileSync(srcFile, encoding);
  fs.writeFileSync(destFile, content, encoding);
}
mongoClient.connect(mongoUri, function (err, db) {
  if (err) {
    return console.error(err);
  }
  // var categoryId = 1;
  // var categoryName = "Flats";
  // var categoryId = 2;
  // var categoryName = "Sandals";
  // var categoryId = 3;
  // var categoryName = "Boots";
  var categoryId = 4;
  var categoryName = "Heels";
	
  var cursor = db.collection('shoes_copy').find({ 
			"shoe.images": { 
				$elemMatch : {
					"_id": { 
						$exists: true 
					}
				}
			},
			"shoe.categories": { 
			  $in : 
			    [
			      "Boots",
			      "Heels",
			      "Sandals",
			      "Flats"
			    ]                  
			}
		},
		{
      // limit: 300
		}
	);
	// var counter = 0;
	cursor.toArray(function(err, results) {
		// cursor.each(function(err, item) {
		for(counter in results) {
			var item = results[counter];
      // console.dir(item);
	    var dest = "../data/category/"  + item.shoe.categories[item.shoe.categories.length-1].replace(/\//g,"_").replace(/ /g,"_") + "/"				
			if (!fs.existsSync(dest)){
				fs.mkdirSync(dest)				  				  
			}
			if (item.shoe.images[0]._id){
				var files = fs.readdirSync(dest);
        copyFileSync('/Users/rdefeo/Development/Other/jemboo/getter/data/images/' + item.shoe.images[0]._id.toString(), dest + files.length + ".jpg")				  
			}
      
      // var file = item.shoe.categories[item.shoe.categories.length-1] + "_" + counter;
      //         var origin = '/Users/rdefeo/Development/Other/jemboo/getter/data/images/' + item.shoe.images[0]._id.toString();
      //         
      //         copyFileSync(origin, dest + file + ".jpg")
			// dest = dest + file + ".jpg"
			//        console.log("writing file " + dest);
			//        
			//        copyFileSync('/Users/rdefeo/Development/Other/jemboo/getter/data/images/' + results[counter].shoe.images[0]._id.toString(), dest)
			//        
			// fs.createReadStream('/Users/rdefeo/Development/Other/jemboo/getter/data/images/' + results[counter].shoe.images[0]._id.toString()).pipe(fs.createWriteStream(dest));
			
			// counter++;
		}
    process.exit(0)

	});
	   
  	
	
  
});
