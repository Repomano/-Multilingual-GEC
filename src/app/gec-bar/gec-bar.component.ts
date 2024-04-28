import  { Component, OnInit, ViewChild, ElementRef }  from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-gec-bar',
  templateUrl: './gec-bar.component.html',
  styleUrls: ['./gec-bar.component.css']
})
export class GecBarComponent implements OnInit{
  title = 'ital-gec';  
  searchQuery = "";
  responseMessage = "";

  constructor(private http: HttpClient) {}
  
  @ViewChild('logo', { static: true }) logoElement!: ElementRef;
  isLogoFinished = false;
  
  ngOnInit() {
    const img = new Image();
    img.onload = () => {
      const logo = this.logoElement.nativeElement;
      logo.style.display = 'block';
      img.classList.add('finished');
      this.isLogoFinished = true;
    };
    img.src = 'assets/Logo.gif';
  }
  
  sendQuery(event: Event) {
    event.preventDefault(); // prevent the form from submitting and reloading the page
    
    const url = 'http://localhost:15000/get_sentence.py'; 
    const body = { query: this.searchQuery };
    this.http.post<any>(url, body).subscribe(
      (response) => {
		console.log(response.message)
        this.responseMessage = response.message;
        console.log(response.message)
      },
      (error) => {
		  this.responseMessage="problem"
		  console.log(this.responseMessage)
		  console.error(error);
		  }
    );
  }
}
